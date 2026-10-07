"""Robust cursor control with search-window (crop-and-zoom) tracking.

Goal: a cursor that is *stable and predictable* even when detection is noisy,
intermittent or briefly wrong -- not a low-latency, high-precision pointer.

Detection is HOW we feed the detector frames, never the model itself:

    REACQUIRE : no known hand -> detect on the FULL frame, as before.
    TRACK     : hand known    -> crop a square window around the last position
                (side = SEARCH_MARGIN x last box width), upscale it to the
                detector's native 256x256, detect there, map results back to
                frame coordinates.

A far hand that is a tiny fraction of the full frame becomes a large fraction
of the window, so the network sees far more pixels of it.  Measured on the
static image with a synthetic far hand, full-frame detection dies at ~0.18 of
frame width while the windowed path still detects down to ~0.033.

Position pipeline (each stage sees only what the previous one accepted):

    detect (full frame | window) -> crop verifier (TRACK only) -> plausibility
        gate -> heavy smoothing -> speed-capped cursor state machine
        (instant lock on a verified position, hold-on-loss)

In TRACK mode a learned hand-vs-clutter classifier (`crop_verifier`) replaces
the in-window box-size gate.  The relaxed 0.04 gate was a deliberate trade that
bought range at the cost of locking onto room clutter; the classifier removes
most of that cost (measured on the recorded clips: 156 clutter locks -> 32 while
keeping 97% of the true hand frames).  `--no-verifier` restores v2 behaviour.

There is no acquisition dwell any more: the "hover the cursor over this square"
mechanic was removed once tracking quality allowed it, so a verified position
locks and starts driving the cursor immediately.  The speed cap and the heavy
smoothing still apply, so acquiring is instant but the cursor still cannot jump.

Run from the project root:
    .venv-blazepalm\\Scripts\\python.exe hand_mouse_cursor.py
Press 'q' in the debug window to quit.
"""
import argparse
import math
import os
import sys
import time
from collections import deque

import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_DEV_ML_DIR = os.path.join(HERE, "blazepalm")
if os.path.isdir(_DEV_ML_DIR):
    sys.path.insert(0, _DEV_ML_DIR)

from blazepalm import PalmDetector                       # noqa: E402
from hand_pipeline import (letterbox, box_to_pixels, hand_roi,   # noqa: E402
                           rotated_rect_to_points, warp_rect,
                           ML_DIR as MODEL_DIR)

try:
    import pyautogui
except ImportError:                                       # pragma: no cover
    pyautogui = None

# Imported at module level (inside try/except rather than lazily) so that a
# frozen build's static analysis definitely bundles crop_verifier.py, while a
# missing module still degrades to the v2 box-size gate instead of crashing.
try:
    from crop_verifier import make_verifier
except ImportError:                                       # pragma: no cover
    make_verifier = None

# ---------------------------------------------------------------------------
# Tunable constants -- edit these by feel.
# ---------------------------------------------------------------------------
# --- cursor state machine -------------------------------------------------
# No acquisition dwell: a verified position locks immediately (the hover-the-
# cursor-over-a-square mechanic was removed once tracking quality allowed it).
MAX_SPEED = 1.0         #     cursor speed cap, in SCREEN WIDTHS per second
LOSS_TIMEOUT = 3.0      # s   no position at all before dropping the cursor lock

# --- detection acceptance -------------------------------------------------
MIN_SCORE = 0.5         # palm-detector score floor
# Full-frame box-size FP gate used while REACQUIREing.  Relaxed 0.15 -> 0.10 in
# v6: a far hand is ~0.085 of the frame width, so at 0.15 it could never be
# acquired at all, and the far "tracking" of v3/v4 was really a lock on static
# clutter big enough to pass the gate with the hand happening to fall inside the
# resulting search window.  The gate can only be relaxed because the acquisition
# MOTION gate now does the clutter rejection the size gate used to do alone.
# Measured (eval_motion_gate.py, integrated, gate ON): at 0.10 FarLight goes
# from 0 to 27 tracked hand frames with 7 clutter locks, and both no-hand clips
# stay at 0 acquisitions -- at every gate down to 0.06.
MIN_BOX_FRAC_REACQUIRE = 0.10
# ...and the pre-v6 value, still used when the motion gate is switched off
# (--no-motion-gate), so that flag remains a genuine rollback: a loose size gate
# without the motion gate would be *weaker* than v4, not equal to it.
MIN_BOX_FRAC_REACQUIRE_NO_MOTION = 0.15
# In TRACK mode the crop VERIFIER replaces the size gate: a learned
# hand-vs-clutter decision is range-independent, whereas the 0.04 size gate is
# exactly what let room clutter (poster, corner) be tracked as a hand.  The
# size gate is kept as a fallback for when no verifier model is available.
MIN_BOX_FRAC_TRACK = 0.04
VERIFIER_THRESHOLD = 0.30   # measured on the real clips against MediaPipe
# ground truth (eval_verifier_vs_gt.py), of 499 true hand frames and 156 clutter
# locks for the 0.04 gate:
#     thr   hand frames kept   clutter locks
#     0.20        490               143
#     0.30        486                32   <- chosen: keeps 97% of the tracking
#     0.40        471                12      and removes 4/5 of the FP locks
#     0.50        461                 9
# Raise it with --verifier-threshold if live testing still shows locks.
VERIFIER_FILENAME = "verifier_cnn.pt"
BLOB_VERIFIER_FILENAME = "blob_verifier.pt"

# --- search window --------------------------------------------------------
SEARCH_MARGIN = 3.0     # window side = margin x last known box width
REACQUIRE_HITS = 2      # consecutive full-frame detections needed before we
                        # COMMIT to a lock.  A single lucky false positive would
                        # otherwise lock us in, and the relaxed in-window gate
                        # would then keep that clutter alive (measured: 3.8% ->
                        # 22.6% accepted frames on the empty dark-room clip).
WIDEN_PER_MISS = 1.4    # window grows this much on each consecutive miss
SHRINK_PER_HIT = 1.0    # on a hit the window snaps straight back to its base
                        # (1.0 = snap).  A gradual shrink does NOT work: with
                        # intermittent detections the per-miss widening outruns
                        # it, the window inflates toward the frame size and the
                        # zoom benefit disappears (measured: median window was
                        # 1058px of a 1920px frame).  A detection just told us
                        # where the hand is, so a tight window is correct again.
MIN_WINDOW_PX = 32.0    # never shrink the window below this
TRACK_LOST_TIME = 3.0   # s of consecutive misses before dropping to REACQUIRE
MAX_WINDOW_FRAC = 1.0   # window side can never exceed this x frame size

# --- multi-pass spatial consensus for REACQUIRE (OFF by default) -----------
# REACQUIRE_HITS above asks for N consecutive detections but says nothing about
# WHERE they are, so a candidate that flickers between the hand and some other
# object can commit on two frames that disagree.  This asks for N consecutive
# accepted passes that also AGREE SPATIALLY, i.e. all inside a small radius:
# "N hits in a row that are all roughly the same place".
#
# Chosen from measurement (diagnose_acquire_consensus.py replays the real dark
# acquisition path over every clip and simulates the rule):
#   N=3  N=4 collapses acquisition on the intermittent clips -- mean delay to the
#        first lock on VeryLowLight went 3 -> 55 frames at N=3 but 160 frames at
#        N=4 (10.7 s at that clip's 15 fps), and total commits fell 341 -> 99 for
#        ~2 further points of false-commit reduction.  N=2 plus a tolerance is
#        barely distinguishable from the shipped behaviour.
#   TOL  genuinely moving hands displace 0.046-0.057 frame widths between
#        consecutive accepted frames (p95 on the far/moving clips), and the
#        hand<->not-hand excursions this is meant to reject start at 0.069-0.074
#        (CloseLightStill, VeryLowLight).  0.05 sits between the two, so a
#        deliberate hand is not broken up while an armpit-scale excursion is.
#        NOT chosen tighter because the false-commit fraction is nearly FLAT in
#        the tolerance (37.9% at 0.02 vs 33.1% at 0.05), so a tighter radius buys
#        no extra certainty and only discards genuine hits.
# MEASURED LIMITATION, stated up front: this rejects TRANSIENT excursions.  It
# cannot reject a false detection that is stable across frames, and on the
# CloseDarkStill clip the dark variant's first lock is wrong at every setting --
# the consensus only delays it (11 -> 15 -> 16 frames).  See DARK_CONSENSUS.md.
ACQUIRE_CONSENSUS_N = 3
ACQUIRE_CONSENSUS_TOL = 0.05   # fraction of frame width

# --- "if it is not certain, do not move the mouse there" -------------------
# Two separate rules, because the cursor was being moved by two different kinds of
# unconfirmed candidate:
#
#   1. REACQUIRE published its FIRST accepted pass.  With --acquire-consensus the
#      LOCK waited for N agreeing passes, but the cursor had already been dragged
#      to pass 1 -- so the flicker still moved the mouse.  Now a position is only
#      published once the run is confirmed.
#   2. TRACK accepted any in-window candidate, so a detection that jumped off the
#      hand moved the cursor immediately and then moved it back ("locks onto an
#      armpit, self-corrects within milliseconds").  Now a candidate further than
#      the tolerance from the committed track is not published at all: the cursor
#      holds exactly still until N consecutive candidates agree on the new place.
#      Measured: this is where the off-hand frames actually come from -- 307 of 317
#      on the hand clips, versus 10 from REACQUIRE.
#
# Both are OFF by default (N = 0), which is v10's behaviour exactly: every accepted
# hit is published immediately.
TRACK_CONSENSUS_N = 3
# Above the genuine frame-to-frame motion of a hand being tracked and below the
# distances at which a candidate jumps off the hand.  Measured p95 of real tracking
# motion: 0.0263 (VeryLowLight), 0.0290 (CloseDarkStill), 0.0332 (FarDark), 0.0354
# (FarLight), 0.0440 (CloseLightMoving), 0.0853 (CloseLightStill -- inflated by the
# evaluation's own 0.06 match radius, since two positions can be 0.09 apart and
# both count as "on the hand").  Measured jump distances: median 0.0742
# (CloseDarkStill), 0.0824 (FarLight), 0.0638 (VeryLowLight), 0.1094
# (CloseLightStill).  0.06 sits above the real motion on five of the six clips and
# below the jumps it is meant to catch; the frames it over-holds on the sixth are
# bounded by TRACK_CONSENSUS_MAX_HOLD below.
TRACK_CONSENSUS_TOL = 0.06     # fraction of frame width
# A hold MUST be bounded.  A hand moving steadily faster than the tolerance
# produces a stream of claims that never agree with each other, so an unbounded
# "hold until N claims agree" would freeze the cursor until the 3 s loss timeout --
# turning a wrong jump into a dead cursor, which is worse.  After this many
# consecutively held frames the claim is accepted anyway, so the rule delays a
# jump rather than vetoing it, and only transient excursions (the reported symptom:
# "self-corrects within milliseconds") are actually suppressed.  At 30 fps this is
# 200 ms; at the 15 fps capture rate 400 ms.
TRACK_CONSENSUS_MAX_HOLD = 6   # frames

# --- plausibility gate ----------------------------------------------------
MAX_HAND_SPEED = 3.0    # frame widths per second. Faster => treated as noise
                        # (= 0.10 frame widths/frame at 30 fps)

# --- motion gate (ACQUISITION ONLY) ---------------------------------------
# "Has this thing ever moved?"  Every remaining false lock (poster, room corner)
# is on a static object, and even a hand held still has micro-tremor, so a
# candidate that has not moved cannot be a hand.  Applied in REACQUIRE only --
# once TRACKing, the CNN verifier alone governs, because a hand that has been
# confirmed is allowed to go still (same logic as hold-on-miss).
#
# ASSUMPTION: the camera is stationary.  If the camera moves, everything in the
# frame moves and this signal says "hand" about the whole room; the global-change
# guard below is what suspends acquisition when that is suspected.  See
# MOTION_GATE_DESIGN.md for the measurements behind every number here.
MOTION_RING = 5          # frames in the differencing ring (~170 ms at 30 fps)
MOTION_WIDTH = 320       # ring is built from a 320-wide INTER_AREA downscale:
                         # averaging kills sensor noise (static room measures
                         # 0.000 changed pixels) while the hand's tremor, being
                         # spatially coherent, survives (0.083 -> 0.078).
MOTION_PIXEL_THR = 12.0  # per-pixel luma change (of 255) counted as "changed"
MOTION_MIN = 0.01        # min FRACTION of changed pixels in the candidate box.
                         # Measured: static room 0.000, weakest still hand
                         # (dark) 0.036, typical still hand 0.078.
MOTION_BG_RATIO = 3.0    # ...and the candidate must also beat the median
MOTION_BG_GRID = 4       # background tile by this factor (self-calibration for
MOTION_MIN_TILES = 4     # a noisy camera); tiles per axis, and how many usable
                         # background tiles are needed before the ratio applies
MOTION_MIN_REGION_FRAC = 0.08   # floor on the motion region size, as a fraction
                                # of frame width, so a far hand is not a speck
GLOBAL_CHANGE_MAX = 3.0  # consecutive-frame |mean luma| change (of 255) above
                         # which acquisition is suspended: auto-exposure or a
                         # moving camera, either way the signal is untrustworthy

# --- motion-blob candidates (OFF by default, A/B via --motion-blobs) ------
# The motion gate above only *rejects* static candidates; the detector's box is
# still the only thing that proposes where the hand is.  Measured on
# VeryLowLight.mp4 that proposal is the dominant failure (64.3% of hand frames
# lost: the detector fires confidently but its box is usually not on the hand),
# while the hand itself is plainly visible in the same frame-difference signal --
# it is moving.  So use that signal to PROPOSE candidate regions too.  Motion
# proposes, the verifier still confirms: every blob must clear the same CNN
# threshold as a detector candidate, so this stays an AND, not an OR.
MOTION_BLOB_PIXEL_THR = 6.0        # separate, LOWER pixel threshold for blob
                                   # PROPOSAL only.  Measured (sweep_motion_
                                   # sensitivity.py): 6/255 recovers slow/subtle
                                   # motion that 12/255 misses -- the held-still
                                   # tremor case goes 36.8% -> 45.8% of frames
                                   # with an on-hand blob, the target clip
                                   # 27.0% -> 31.5% -- and it is only safe
                                   # because the area floor below moves with it.
                                   # The gate's own 12/255 metric is unchanged:
                                   # it was validated separately and still is.
MOTION_BLOB_MIN_SIDE_FRAC = 0.05   # smallest blob worth judging, as a fraction
                                   # of frame width; the code squares it, so this
                                   # is an AREA floor of (0.05 x width)^2.
                                   # Raised 0.03 -> 0.05 alongside the lower pixel
                                   # threshold, and that is what keeps the noise
                                   # cost at zero: noise speckle clusters are
                                   # small, a slowly moving hand is not.  At 0.03
                                   # the lower threshold lets static-room false
                                   # blobs through (2.7% -> 10.8% of frames); at
                                   # 0.05 the no-hand clips stay at 0.0%.
MOTION_BLOB_MAX = 3                # judge at most this many blobs per frame
MOTION_BLOB_ROI_SCALE = 1.2        # blob bbox -> the pipeline's 2.6x-ROI
                                   # convention.  A translating hand's motion
                                   # blob is roughly the hand, not the ROI, so
                                   # this converts between them: measured ratio
                                   # (2.6 x palm width / blob side) has median
                                   # 1.20 over 317 blob-on-hand pairs
                                   # (calibrate_motion_blobs.py).  Chosen on
                                   # geometry, NOT on which value scored best --
                                   # no value scored well.
MOTION_BLOB_ROTATIONS = 8          # in-plane orientations searched per blob: a
                                   # blob has no keypoints, so the wrist->MCP
                                   # rotation the verifier was trained with is
                                   # unknown.  Measured tolerance in
                                   # calibrate_motion_blobs.py
MOTION_BLOB_PREFER_MARGIN = 0.10   # in TRACK, a blob only overrides the
                                   # detector's own candidate if it beats it by
                                   # this much P(hand), to protect the existing
                                   # continuity behaviour
MOTION_BLOB_THRESHOLD = 0.80       # threshold for the BLOB-specific verifier,
                                   # which is a different distribution from the
                                   # detector's and needs its own operating point.
                                   # Measured on held-out blobs (sweep_blob_
                                   # threshold.py): at 0.30 the blob verifier
                                   # accepts 54.7% of moving clutter, which makes
                                   # the pipeline lock on clutter 35 times in 45
                                   # frames; at 0.80 it accepts 75.8% of real hand
                                   # blobs and only 5.6% of clutter -- 3.3x the
                                   # hand recall of the main verifier at LOWER
                                   # clutter acceptance (22.6%/9.5% at 0.30).

# --- smoothing (deliberately heavy; the point is a boring cursor) ---------
SMOOTHING_TAU = 0.35    # s. EMA time constant (~0.45 Hz low-pass). Heavier
                        # than the old One Euro at rest (~0.17/frame vs ~0.09)

REACQUIRE = "REACQUIRE"
TRACK = "TRACK"
SEARCHING = "SEARCHING"
LOCKED = "LOCKED"


# ---------------------------------------------------------------------------
# search-window geometry
# ---------------------------------------------------------------------------
def crop_window(frame, cx, cy, side):
    """Square crop of `side` px centred on (cx, cy), clamped inside the frame.

    Returns (crop, x0, y0, side) -- the returned side is the actual integer side
    used, and (x0, y0) is the crop's top-left in frame pixels.
    """
    h, w = frame.shape[:2]
    side = int(max(16, min(int(round(side)), min(h, w))))
    x0 = max(0, min(int(round(cx - side / 2.0)), w - side))
    y0 = max(0, min(int(round(cy - side / 2.0)), h - side))
    return frame[y0:y0 + side, x0:x0 + side], x0, y0, side


def map_crop_box_to_frame(box_norm, x0, y0, side):
    """Detector box (normalised [0,1] in the square crop) -> frame pixels.

    The crop is a square resized uniformly to 256x256, so this is a single
    scale factor plus the crop origin -- no aspect padding is involved.
    """
    return (x0 + float(box_norm[0]) * side, y0 + float(box_norm[1]) * side,
            x0 + float(box_norm[2]) * side, y0 + float(box_norm[3]) * side)


def detect_full(detector, frame, min_box_frac=MIN_BOX_FRAC_REACQUIRE):
    """Detections on the whole frame (letterboxed to 256), as in v1.

    Returns (box, score, p_hand) triples; p_hand is None here because the
    full-frame size gate is left exactly as it was in v1/v2 -- see the note on
    MIN_BOX_FRAC_REACQUIRE.  The verifier only guards the TRACK window, which is
    the mode whose relaxed gate was admitting clutter.
    """
    h, w = frame.shape[:2]
    padded, scale, left, top = letterbox(frame, 256)
    with torch.no_grad():
        dets = detector.predict_on_image(padded)
    out = []
    if not dets:
        return out
    for d in dets[0]:
        score = float(d[18])
        if score < MIN_SCORE:
            continue
        box = box_to_pixels(d[:4].tolist(), scale, left, top)
        if (box[2] - box[0]) < min_box_frac * w:
            continue
        out.append((box, score, None))
    return out


def detect_window(detector, frame, cx, cy, side,
                  min_box_frac=MIN_BOX_FRAC_TRACK,
                  verifier=None, verifier_threshold=VERIFIER_THRESHOLD,
                  stats=None):
    """Detections inside a search window, mapped back to frame pixels.

    With a `verifier` the learned hand-vs-clutter classifier replaces the box
    size gate entirely: each candidate is judged on its OWN rotation-normalised
    crop, rebuilt from the frame with exactly the recipe the classifier was
    trained on (hand_roi from the decoded box + its two keypoints -> rotated
    rect -> warp).  That decision is range-independent, which is the property
    the size gate cannot have -- and the size gate is precisely what let room
    clutter (a poster, a corner) be tracked at distance.

    `stats`, if given, is a list that receives the p_hand of every REJECTED
    candidate (debug/telemetry only).

    Returns (box, score, p_hand) triples; p_hand is None on the fallback path.
    """
    h, w = frame.shape[:2]
    crop, x0, y0, side = crop_window(frame, cx, cy, side)
    inp = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
    with torch.no_grad():
        dets = detector.predict_on_image(inp)
    out = []
    if not dets:
        return out
    for d in dets[0]:
        score = float(d[18])
        if score < MIN_SCORE:
            continue
        box = map_crop_box_to_frame(d[:4].tolist(), x0, y0, side)
        if verifier is None:
            if (box[2] - box[0]) < min_box_frac * w:  # still in frame pixels
                continue
            out.append((box, score, None))
            continue
        # Keypoints are normalised to the square crop, which was resized
        # uniformly to 256 -- so mapping back is the same scale plus origin.
        kp0 = (x0 + float(d[4]) * side, y0 + float(d[5]) * side)
        kp2 = (x0 + float(d[8]) * side, y0 + float(d[9]) * side)
        hx, hy, hs, hr = hand_roi(box, kp0, kp2)
        rect = rotated_rect_to_points(hx, hy, hs, hs, hr)
        hand_crop, _ = warp_rect(frame, rect, 256)
        p = float(verifier(hand_crop))
        if p < verifier_threshold:
            if stats is not None:
                stats.append(p)
            continue
        out.append((box, score, p))
    return out


# ---------------------------------------------------------------------------
# motion gate (acquisition only)
# ---------------------------------------------------------------------------
class MotionGate:
    """Frame-differencing "has this region moved recently?" test.

    Keeps a ring of the last MOTION_RING frames as 320-wide luma (INTER_AREA),
    and scores a candidate box by the FRACTION of its pixels whose
    bias-compensated luma changed by more than MOTION_PIXEL_THR between the
    oldest and newest ring frame.

    Why the fraction and not the mean difference: measured on our clips, a hand
    held still (0.036-0.078) and a static room (0.000) are separable by the
    changed-pixel fraction, but their mean absolute differences overlap or
    invert (0.79-1.89x) -- a naive |frame - frame| average cannot tell a still
    hand from a poster.  See MOTION_GATE_DESIGN.md.

    Two robustness measures are built in:

      * frame-wide gain compensation -- the old frame is rescaled to the new
        frame's global mean/std before differencing, so a global brightness
        change (auto-exposure) is divided out;
      * background self-calibration -- the candidate must beat the median
        background tile by MOTION_BG_RATIO (tiles overlapping the candidate are
        excluded), so a camera or a scene that is noisy everywhere raises the
        bar instead of silently passing everything.

    The gate is fail-closed: until the ring is full it reports "no motion
    evidence", which only delays acquisition by ~170 ms.
    """

    def __init__(self, ring=MOTION_RING, width=MOTION_WIDTH,
                 pixel_thr=MOTION_PIXEL_THR, min_frac=MOTION_MIN,
                 bg_ratio=MOTION_BG_RATIO, bg_grid=MOTION_BG_GRID,
                 min_tiles=MOTION_MIN_TILES,
                 min_region_frac=MOTION_MIN_REGION_FRAC,
                 global_max=GLOBAL_CHANGE_MAX):
        self.ring = int(ring)
        self.width = int(width)
        self.pixel_thr = float(pixel_thr)
        self.min_frac = float(min_frac)
        self.bg_ratio = float(bg_ratio)
        self.bg_grid = int(bg_grid)
        self.min_tiles = int(min_tiles)
        self.min_region_frac = float(min_region_frac)
        self.global_max = float(global_max)
        self.frames = deque(maxlen=self.ring)
        self.prev_mean = None
        self.gshift = 0.0            # |mean luma change| vs the previous frame
        self._diff = None            # cached boolean diff for the current ring
        self._blob_diff = None       # same, at the lower blob-proposal threshold

    # -- ingest -----------------------------------------------------------
    def push(self, frame_bgr):
        """Add one frame.  Must be called once per frame, in order."""
        fh, fw = frame_bgr.shape[:2]
        scale = self.width / float(fw)
        ch = max(2, int(round(fh * scale)))
        small = cv2.resize(frame_bgr, (self.width, ch),
                           interpolation=cv2.INTER_AREA)
        lu = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = float(lu.mean())
        # consecutive frames, not ring endpoints: an alternating auto-exposure
        # step is invisible to an oldest-vs-newest comparison.
        self.gshift = abs(m - self.prev_mean) if self.prev_mean is not None else 0.0
        self.prev_mean = m
        self.frames.append(lu)
        self._diff = None
        self._blob_diff = None

    def ready(self):
        return len(self.frames) == self.frames.maxlen

    def suspended(self):
        """True when the frame changed globally: AE event or moving camera."""
        return self.gshift > self.global_max

    # -- internals --------------------------------------------------------
    def _ensure_diff(self, blob=False):
        """Difference image of the ring, at the gate or the blob threshold."""
        attr = "_blob_diff" if blob else "_diff"
        if getattr(self, attr) is not None or not self.ready():
            return
        new, old = self.frames[-1], self.frames[0]
        # divide out a frame-wide brightness/gain change before differencing
        g = float(new.std() / (old.std() + 1e-6))
        old = g * old + (float(new.mean()) - g * float(old.mean()))
        a = new - new.mean()
        b = old - old.mean()
        thr = MOTION_BLOB_PIXEL_THR if blob else self.pixel_thr
        setattr(self, attr, np.abs(a - b) > thr)

    def _tiles(self):
        """Per-tile changed fractions as a (gy, gx) array, or None."""
        d = self._diff
        h, w = d.shape
        gy = gx = self.bg_grid
        th, tw = h // gy, w // gx
        if th < 1 or tw < 1:
            return None
        return d[:th * gy, :tw * gx].reshape(gy, th, gx, tw).mean(axis=(1, 3))

    def background_frac(self, region):
        """Median changed fraction of the background tiles.

        Tiles whose centre falls inside the candidate region are excluded, so a
        big close-up hand cannot drag the background estimate up and lock
        acquisition out.  Returns None when too few tiles remain to calibrate.
        """
        tiles = self._tiles()
        if tiles is None:
            return None
        gy, gx = tiles.shape
        h, w = self._diff.shape
        th, tw = h / gy, w / gx
        cx, cy, side = region
        keep = []
        for i in range(gy):
            for j in range(gx):
                tx, ty = (j + 0.5) * tw, (i + 0.5) * th
                if abs(tx - cx) > side / 2.0 or abs(ty - cy) > side / 2.0:
                    keep.append(tiles[i, j])
        if len(keep) < self.min_tiles:
            return None
        return float(np.median(keep))

    # -- candidate PROPOSAL (motion blobs) --------------------------------
    def blobs(self, frame_shape, min_side_frac=MOTION_BLOB_MIN_SIDE_FRAC,
              max_blobs=MOTION_BLOB_MAX):
        """Connected regions of changed pixels, as candidate boxes in frame px.

        This is the second, detector-independent way to say WHERE the hand is:
        the same difference signal that the gate uses to reject static
        candidates also shows a moving hand clearly.  Returns a list of
        (cx, cy, side, area_frac) in FRAME pixels, largest first.
        """
        if not self.ready():
            return []
        self._ensure_diff(blob=True)
        fh, fw = frame_shape[:2]
        mask = self._blob_diff.astype(np.uint8)
        n_lab, _labels, stats, cents = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        scale = float(fw) / self.width
        min_side = min_side_frac * self.width
        min_area = min_side * min_side
        out = []
        for i in range(1, n_lab):                       # 0 is background
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            side = max(w, h)
            # grow the bbox a little: the blob is the moving part of the hand,
            # usually smaller than the hand itself
            side = side * 1.15
            out.append((float(cents[i][0]), float(cents[i][1]),
                        float(side), float(area) / float(mask.size)))
        out.sort(key=lambda b: -b[3])
        out = out[:max_blobs]
        # ring coordinates -> frame pixels
        return [(cx * scale, cy * scale, max(side * scale, 8.0), af)
                for cx, cy, side, af in out]

    def score(self, box_px, frame_shape):
        """Score a candidate box.  Returns (ok, info dict)."""
        if not self.ready():
            return False, dict(frac=None, bg=None, threshold=self.min_frac,
                               reason="ring warming up")
        self._ensure_diff()
        fh, fw = frame_shape[:2]
        scale = self.width / float(fw)
        x0, x1 = sorted((box_px[0] * scale, box_px[2] * scale))
        y0, y1 = sorted((box_px[1] * scale, box_px[3] * scale))
        # square region of side max(box width, floor) centred on the box
        h, w = self._diff.shape
        side = max(x1 - x0, self.min_region_frac * self.width)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx0 = int(max(0, min(round(cx - side / 2.0), w - 1)))
        rx1 = int(max(rx0 + 1, min(round(cx + side / 2.0), w)))
        ry0 = int(max(0, min(round(cy - side / 2.0), h - 1)))
        ry1 = int(max(ry0 + 1, min(round(cy + side / 2.0), h)))
        frac = float(self._diff[ry0:ry1, rx0:rx1].mean())
        region = ((rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0, float(rx1 - rx0))
        bg = self.background_frac(region)
        threshold = self.min_frac
        reason = ""
        if bg is not None and self.bg_ratio * bg > threshold:
            threshold = self.bg_ratio * bg
            reason = f"background is live ({bg:.3f})"
        ok = frac >= threshold
        if ok and self.suspended():
            ok = False
            reason = f"global change {self.gshift:.1f}"
        return ok, dict(frac=frac, bg=bg, threshold=threshold, reason=reason)


# ---------------------------------------------------------------------------
# motion-blob candidates: propose with motion, confirm with the verifier
# ---------------------------------------------------------------------------
def best_rotation_score(frame, cx, cy, side, verifier,
                        rotations=MOTION_BLOB_ROTATIONS):
    """Best P(hand) over in-plane orientations of a square ROI.

    Needed because neither a motion blob nor a bare detector box carries the
    wrist->MCP keypoints the normal crop recipe uses.  Applied to BOTH candidate
    sources in the motion-blob path so their scores are comparable.
    """
    best_p, best_rot = -1.0, 0.0
    for k in range(rotations):
        rot = math.pi * k / rotations
        rect = rotated_rect_to_points(cx, cy, side, side, rot)
        crop, _ = warp_rect(frame, rect, 256)
        p = float(verifier(crop))
        if p > best_p:
            best_p, best_rot = p, rot
    return best_p, best_rot


def blob_candidates(frame, gate, verifier, verifier_threshold=VERIFIER_THRESHOLD,
                    roi_scale=MOTION_BLOB_ROI_SCALE,
                    rotations=MOTION_BLOB_ROTATIONS, stats=None,
                    within=None):
    """Turn moving regions into verifier-confirmed candidate positions.

    A motion blob carries no keypoints, so the wrist->MCP rotation the verifier
    was trained with is unknown: `rotations` in-plane orientations are tried and
    the best P(hand) wins.  Nothing is accepted on motion alone -- a blob that
    the verifier rejects is discarded exactly like a detector box would be.

    `within`, if given, is (cx, cy, side) and restricts proposals to blobs inside
    that window (used by TRACK).

    Returns a list of (box_px, p_hand, (cx, cy, side, area_frac)) sorted by
    descending p_hand.
    """
    if verifier is None:
        # Unconfirmed motion proposals are exactly the unsafe mode: measured to
        # accept 88-94% of OFF-hand blobs.  Refuse rather than crash or guess.
        return []
    out = []
    for cx, cy, side, area_frac in gate.blobs(frame.shape):
        r = side * roi_scale
        if within is not None:
            wx, wy, wside = within
            if (abs(cx - wx) > wside / 2.0 + r / 2.0
                    or abs(cy - wy) > wside / 2.0 + r / 2.0):
                continue
        best_p, _rot = best_rotation_score(frame, cx, cy, r, verifier, rotations)
        if stats is not None:
            stats.append(best_p)
        if best_p < verifier_threshold:
            continue
        box = (cx - r / 2.0, cy - r / 2.0, cx + r / 2.0, cy + r / 2.0)
        out.append((box, best_p, (cx, cy, r, area_frac)))
    out.sort(key=lambda t: -t[1])
    return out


# ---------------------------------------------------------------------------
# plausibility gate
# ---------------------------------------------------------------------------
class PlausibilityGate:
    """Reject positions implying a faster-than-possible jump since the last
    ACCEPTED position.  A rejected position changes nothing: it is not a hit
    and does not become the new reference.
    """

    def __init__(self, max_speed_frac=MAX_HAND_SPEED, dt_cap=1.0):
        self.max_speed_frac = float(max_speed_frac)
        self.dt_cap = float(dt_cap)
        self.pos = None
        self.t = None

    def check(self, pos, t, frame_w):
        pos = np.asarray(pos, dtype=float)
        if self.pos is None:                     # nothing to compare against
            self.pos, self.t = pos, t
            return True
        dt = min(max(float(t) - float(self.t), 1.0 / 240.0), self.dt_cap)
        limit = self.max_speed_frac * float(frame_w) * dt
        if float(np.linalg.norm(pos - self.pos)) > limit:
            return False                         # implausible -> discard
        self.pos, self.t = pos, t
        return True

    def reset(self):
        self.pos = None
        self.t = None


# ---------------------------------------------------------------------------
# heavy smoothing
# ---------------------------------------------------------------------------
class PositionSmoother:
    """Deliberately slow exponential moving average (screen pixels).

    alpha = 1 - exp(-dt/tau), so the time constant is frame-rate independent.
    Holds its value when given None (a detection gap), so a gap cannot drag it.
    """

    def __init__(self, tau=SMOOTHING_TAU):
        self.tau = float(tau)
        self.value = None

    def update(self, pos, dt):
        if pos is None:
            return None if self.value is None else self.value.copy()
        pos = np.asarray(pos, dtype=float)
        if self.value is None:
            self.value = pos.copy()
            return self.value.copy()
        alpha = 1.0 - math.exp(-max(float(dt), 0.0) / self.tau) if self.tau > 0 else 1.0
        self.value = self.value + alpha * (pos - self.value)
        return self.value.copy()

    def reset(self):
        self.value = None


# ---------------------------------------------------------------------------
# detector front-end: REACQUIRE / TRACK
# ---------------------------------------------------------------------------
class HandLocator:
    """Two-mode locator.  Owns the search window, the miss counter, the
    plausibility gate and (since v5) the acquisition-only motion gate; returns
    an accepted frame-pixel position or None.
    """

    def __init__(self, detector, margin=SEARCH_MARGIN, gate=None,
                 lost_time=TRACK_LOST_TIME, verifier=None,
                 verifier_threshold=VERIFIER_THRESHOLD, motion_gate=None,
                 reacquire_min_box_frac=None, use_motion_blobs=False,
                 blob_verifier=None, acquire_consensus=0,
                 acquire_consensus_tol=ACQUIRE_CONSENSUS_TOL,
                 track_consensus=0,
                 track_consensus_tol=TRACK_CONSENSUS_TOL):
        self.detector = detector
        self.margin = float(margin)
        self.gate = gate if gate is not None else PlausibilityGate()
        self.lost_time = float(lost_time)
        self.verifier = verifier
        self.verifier_threshold = float(verifier_threshold)
        # Acquisition-only: passed in explicitly so `motion_gate=None` is a
        # complete rollback to pre-v5 behaviour.
        self.motion_gate = motion_gate
        # Motion-blob candidates (v8, OFF by default).  Enabling this also turns
        # the verifier on during REACQUIRE, because the whole point is to let a
        # verifier-confirmed candidate acquire without a detector box.
        # MEASURED WARNING: in the very lighting this was built for, the verifier
        # does NOT separate on-hand from off-hand motion blobs (see
        # MOTION_BLOBS_DESIGN.md), so this mode proposes well and confirms badly.
        self.use_motion_blobs = bool(use_motion_blobs)
        # Blob crops are their own distribution (measured: for the same hands,
        # blob-pipeline crops score 1.95x the training-style crops the main
        # verifier was trained on), so a blob-specific verifier -- trained on
        # crops built through the blob pipeline itself -- is used for blobs when
        # one is supplied, at its own threshold.
        self.blob_verifier = blob_verifier
        self.blob_rotations = MOTION_BLOB_ROTATIONS
        self.blob_threshold = (MOTION_BLOB_THRESHOLD if blob_verifier is not None
                               else float(verifier_threshold))
        if reacquire_min_box_frac is None:
            # The relaxed acquisition size gate is only justified because the
            # motion gate rejects the static clutter it would otherwise admit,
            # so the two are tied together: no motion gate -> v4's 0.15.
            reacquire_min_box_frac = (MIN_BOX_FRAC_REACQUIRE
                                      if motion_gate is not None
                                      else MIN_BOX_FRAC_REACQUIRE_NO_MOTION)
        self.reacquire_min_box_frac = float(reacquire_min_box_frac)
        # Multi-pass spatial consensus for REACQUIRE only (0 = off, i.e. v10's
        # REACQUIRE_HITS behaviour, which stays the default everywhere else).
        # TRACK deliberately keeps its existing continuity logic: once a real lock
        # exists, holding and smoothing is a different and already-adequate
        # mechanism, and demanding sustained agreement there would make an
        # established cursor stutter.
        self.acquire_consensus = int(acquire_consensus)
        self.acquire_consensus_tol = float(acquire_consensus_tol)
        # "Not certain -> do not move the cursor there": an in-window candidate that
        # jumps away from the committed track is held back until N consecutive
        # candidates agree on the new location.  0 = off (v10 behaviour: publish
        # every accepted hit immediately).
        self.track_consensus = int(track_consensus)
        self.track_consensus_tol = float(track_consensus_tol)
        self.track_consensus_max_hold = int(TRACK_CONSENSUS_MAX_HOLD)
        self.reset()

    def reset(self):
        self.mode = REACQUIRE
        self.pos = None
        self.box_w = None
        self.window = None
        self.misses = 0
        self.miss_time = 0.0
        self.reacquire_hits = 0
        self.consensus_run = []    # positions of the current agreeing run
        self.consensus_resets = 0  # times a candidate broke a run (telemetry)
        self.jump_run = []         # in-window candidates claiming a new location
        self.jump_held = 0         # consecutive frames the claim has been held
        self.jumps_rejected = 0    # times a TRACK jump claim was held back
        self.fw = None             # frame width of the last update, for tolerances
        self.p_hand = None
        self.rejected = []         # p_hand of clutter rejected in the last frame
        self.motion = None         # info dict from the last motion decision
        self.motion_rejected = 0   # candidates in the last frame killed by motion
        self.motion_blobs_unavailable = False
        self.blob_stats = []       # P(hand) of every blob judged this frame
        self.n_blobs = 0           # blobs that cleared the verifier this frame
        self.locked_on_blob = False
        self.gate.reset()

    def info(self):
        m = self.motion or {}
        return dict(mode=self.mode, misses=self.misses,
                    miss_time=self.miss_time, window=self.window,
                    box_w=self.box_w, reacquire_hits=self.reacquire_hits,
                    p_hand=self.p_hand, n_rejected=len(self.rejected),
                    verifier=self.verifier is not None,
                    motion_gate=self.motion_gate is not None,
                    motion_frac=m.get("frac"), motion_bg=m.get("bg"),
                    motion_threshold=m.get("threshold"),
                    motion_reason=m.get("reason", ""),
                    n_motion_rejected=self.motion_rejected,
                    motion_blobs=self.use_motion_blobs,
                    n_blobs=self.n_blobs,
                    n_blobs_judged=len(self.blob_stats),
                    best_blob_p=(max(self.blob_stats) if self.blob_stats else None),
                    locked_on_blob=self.locked_on_blob,
                    acquire_consensus=self.acquire_consensus,
                    consensus_run=len(self.consensus_run),
                    consensus_resets=self.consensus_resets,
                    track_consensus=self.track_consensus,
                    jump_run=len(self.jump_run),
                    jump_held=self.jump_held,
                    jumps_rejected=self.jumps_rejected)

    def update(self, frame, t, dt):
        """Returns (accepted_position_frame_px | None, info)."""
        fh, fw = frame.shape[:2]
        self.fw = fw           # the consensus tolerance is a fraction of this
        if self.motion_gate is not None:
            self.motion_gate.push(frame)

        if self.mode == TRACK and self.pos is not None and self.window:
            self.rejected = []      # p_hand of each clutter rejection, this frame
            dets = detect_window(self.detector, frame, self.pos[0], self.pos[1],
                                 self.window, verifier=self.verifier,
                                 verifier_threshold=self.verifier_threshold,
                                 stats=self.rejected)
            # TRACK: the motion requirement is dropped on purpose -- a hand that
            # has been confirmed is allowed to go still.  The CNN verifier is
            # the gate here.
            self.motion = None
            self.motion_rejected = 0
        else:
            dets = detect_full(self.detector, frame,
                               min_box_frac=self.reacquire_min_box_frac)
            self.rejected = []
            if self.motion_gate is not None:
                dets = self._apply_motion_gate(dets, frame)
            else:
                self.motion = None
                self.motion_rejected = 0

        cand, cand_w, cand_p = None, None, None
        if self.use_motion_blobs and self.motion_gate is not None:
            # ---- v8 motion-blob path (flag-gated; see MOTION_BLOBS_DESIGN.md)
            # Both candidate sources are scored the SAME way (rotation search over
            # the box) so they are comparable, and only a candidate that clears
            # the verifier threshold can be used at all -- motion proposes, the
            # verifier confirms, same AND discipline as everywhere else.
            self.blob_stats = []
            self.locked_on_blob = False
            if dets and self.verifier is None:
                # No verifier at all: fall through to the plain non-blob selection
                # rather than calling None (this was the HandCursorDark_v1 crash).
                dets = dets
                self.motion_blobs_unavailable = True
            elif dets:
                scored = []
                for d in dets:
                    b = d[0]
                    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                    p, _ = best_rotation_score(frame, bcx, bcy, b[2] - b[0],
                                               self.verifier)
                    scored.append((b, p))
                if self.mode == TRACK and self.pos is not None:
                    # keep the continuity rule for the detector's own candidates
                    box, det_p = min(scored, key=lambda t: float(np.linalg.norm(
                        np.array([(t[0][0] + t[0][2]) / 2.0,
                                  (t[0][1] + t[0][3]) / 2.0]) - self.pos)))
                else:
                    box, det_p = max(scored, key=lambda t: t[0][2] - t[0][0])
                if det_p >= self.verifier_threshold:
                    cand = np.array([(box[0] + box[2]) / 2.0,
                                     (box[1] + box[3]) / 2.0])
                    cand_w, cand_p = box[2] - box[0], det_p
            elif dets:
                if self.mode == TRACK and self.pos is not None:
                    def _c(b):
                        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
                    box, _s, cand_p = min(dets, key=lambda t: float(
                        np.linalg.norm(_c(t[0]) - self.pos)))
                else:
                    box, _s, cand_p = max(dets, key=lambda t: t[0][2] - t[0][0])
                cand = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
                cand_w = box[2] - box[0]
            within = None
            if self.mode == TRACK and self.pos is not None and self.window:
                within = (self.pos[0], self.pos[1], self.window)
            blobs = blob_candidates(frame, self.motion_gate,
                                    self.blob_verifier or self.verifier,
                                    self.blob_threshold,
                                    rotations=self.blob_rotations,
                                    stats=self.blob_stats, within=within)
            self.n_blobs = len(blobs)
            if blobs:
                bb, bp, _binfo = blobs[0]
                b_centre = np.array([(bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0])
                if cand is None or bp > (cand_p or 0.0) + MOTION_BLOB_PREFER_MARGIN:
                    cand, cand_w, cand_p = b_centre, bb[2] - bb[0], bp
                    self.locked_on_blob = True
        elif dets:
            if self.mode == TRACK and self.pos is not None:
                # Continuity: take the detection nearest the last known
                # position.  Picking the LARGEST instead is wrong in a window
                # that happens to contain a bigger object (e.g. a second hand
                # or nearby clutter) -- it would lock onto that instead.
                def _centre(d):
                    b = d[0]
                    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
                box, _score, cand_p = min(dets, key=lambda d: float(
                    np.linalg.norm(_centre(d) - self.pos)))
            else:
                # Full-frame search: largest box = most hand-like after the gate
                box, _score, cand_p = max(dets, key=lambda d: d[0][2] - d[0][0])
            cand = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            cand_w = box[2] - box[0]

        if cand is not None and self.gate.check(cand, t, fw):
            self.p_hand = cand_p
            if self._on_hit(cand, cand_w):
                return cand, self.info()
            # The candidate cleared every gate but has NOT been confirmed (an
            # un-agreed REACQUIRE run, or an in-window jump claim).  Publishing it
            # is what dragged the cursor onto the armpit, so publish nothing: the
            # cursor holds exactly where it is.
            if self.mode == TRACK:
                # ...but an unconfirmed jump still ages the lock: if the new place
                # never confirms, the miss timeout eventually re-acquires.
                self._on_miss(dt, fw, fh, hard=False)
            return None, self.info()

        self.p_hand = None
        self._on_miss(dt, fw, fh)          # includes implausible detections
        return None, self.info()

    # -- acquisition motion gate ------------------------------------------
    def _apply_motion_gate(self, dets, frame):
        """AND the motion gate onto full-frame (REACQUIRE) candidates.

        Runs after the size gate inside `detect_full` and before anything else,
        so a static candidate is dropped without any further work.  The best
        motion score seen this frame is kept for the overlay, because "why
        didn't it acquire?" is otherwise invisible.
        """
        self.motion_rejected = 0
        if not dets:
            self.motion = None
            return dets
        keep, best = [], None
        for d in dets:
            ok, info = self.motion_gate.score(d[0], frame.shape)
            if best is None or (info.get("frac") or 0.0) > (best.get("frac") or 0.0):
                best = info
            if ok:
                keep.append(d)
            else:
                self.motion_rejected += 1
        self.motion = best
        return keep

    # -- internal ---------------------------------------------------------
    def _on_hit(self, pos, box_w):
        """Record an accepted candidate.  Returns True if it may be PUBLISHED.

        Publishing is what moves the cursor, so this is the point where "if it is
        not certain, do not move the mouse there" is enforced.  A candidate that
        cleared every gate is not necessarily one we should act on yet: with the
        consensus flags on, an unconfirmed one is withheld and the cursor holds
        exactly still (CursorController does nothing while it has no target).

        With both consensuses off -- v10 and every other build -- every accepted
        hit returns True, so behaviour is byte-for-byte what it always was.
        """
        prev_pos = self.pos      # the COMMITTED position, captured before self.pos
                                 # is touched: the TRACK jump test compares against it
        if self.mode == REACQUIRE:
            if self.acquire_consensus > 1:
                if self._consensus_agrees(pos):
                    self.consensus_run.append(np.asarray(pos, dtype=float))
                else:
                    # Start a fresh run with THIS candidate.  Keeping it means a
                    # single stray frame cannot discard a good sequence that is
                    # just beginning, while an alternating hand/other flicker
                    # still never reaches N (every other frame restarts the run).
                    self.consensus_run = [np.asarray(pos, dtype=float)]
                    self.consensus_resets += 1
                if len(self.consensus_run) < self.acquire_consensus:
                    # NOT confirmed: do not publish, and do not count it as a miss
                    # either -- the run must survive for the next pass to extend it.
                    return False
                self.consensus_run = []
                self.reacquire_hits = 0
            else:
                # v10 path, unchanged: the candidate is published immediately and
                # only the LOCK waits for REACQUIRE_HITS.
                self.pos, self.box_w = pos, box_w
                self.misses = 0
                self.miss_time = 0.0
                self.reacquire_hits += 1
                if self.reacquire_hits < REACQUIRE_HITS:
                    return True
                self.reacquire_hits = 0
            self.pos, self.box_w = pos, box_w
            self.misses = 0
            self.miss_time = 0.0
            self.window = self.margin * box_w          # establish the lock
            self.mode = TRACK
            return True

        # ---- TRACK ---------------------------------------------------------
        # The motion requirement is dropped here on purpose (a confirmed hand may
        # go still), so this is where a wrong in-window detection moves the cursor.
        if self.track_consensus > 1 and prev_pos is not None and self.fw:
            d = float(np.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]))
            if d > self.track_consensus_tol * float(self.fw):
                # This candidate claims the hand is somewhere new.  Hold the cursor
                # where it is until that claim is repeated and agreed.
                self.jump_run.append(np.asarray(pos, dtype=float))
                self.jump_held += 1
                if (len(self.jump_run) >= self.track_consensus
                        and self._jump_agrees()):
                    # A sustained, self-consistent new location: move there.
                    pos = np.mean(self.jump_run, axis=0)
                    self.jump_run = []
                    self.jump_held = 0
                elif self.jump_held >= self.track_consensus_max_hold:
                    # Bounded hold: a hand that really moved must not be frozen out
                    # for the whole 3 s loss timeout.  Accept the claim as-is.
                    self.jump_run = []
                    self.jump_held = 0
                else:
                    self.jumps_rejected += 1
                    return False
            else:
                self.jump_run = []
                self.jump_held = 0
        self.pos, self.box_w = pos, box_w
        self.misses = 0
        self.miss_time = 0.0
        # A detection just told us where the hand is, so snap the window straight
        # back to the tight base -- that restores the zoom which is the whole point
        # of tracking in a window.
        self.window = max(self.margin * box_w,
                          MIN_WINDOW_PX,
                          self.window * SHRINK_PER_HIT)
        return True

    def _consensus_agrees(self, pos):
        """Is this candidate roughly where the current run already is?

        Measured against the RUN MEAN rather than the previous candidate, because
        the requirement the user asked for is "all roughly the same place", not
        "each step is small" -- the latter lets a slow drift walk anywhere.
        """
        if not self.consensus_run or not self.fw:
            return True
        centre = np.mean(self.consensus_run, axis=0)
        d = float(np.hypot(pos[0] - centre[0], pos[1] - centre[1]))
        return d <= self.acquire_consensus_tol * float(self.fw)

    def _jump_agrees(self):
        """Do the last N in-window jump claims all point at the same new place?

        Consistency is judged among the RECENT claims only.  A steadily moving hand
        fails this on purpose: its claims describe a path, not a place, which is
        why the hold has to be bounded by TRACK_CONSENSUS_MAX_HOLD.
        """
        if not self.jump_run or not self.fw:
            return True
        recent = self.jump_run[-self.track_consensus:]
        centre = np.mean(recent, axis=0)
        return all(float(np.hypot(p[0] - centre[0], p[1] - centre[1]))
                   <= self.track_consensus_tol * float(self.fw) for p in recent)

    def _on_miss(self, dt, fw, fh, hard=True):
        self.misses += 1
        self.reacquire_hits = 0                        # break the confirmation run
        # A miss breaks a consensus run too: the passes must be consecutive, so a
        # candidate that appears, vanishes and reappears does not add up to a
        # sustained single location.
        #
        # `hard=False` is the WITHHELD case -- a candidate existed but was not
        # published, which is precisely the run that must be allowed to continue
        # accumulating.  It still ages the lock towards REACQUIRE and still widens
        # the window (the hand may genuinely have moved out of it), but it must not
        # wipe the evidence.
        if hard:
            self.consensus_run = []
            self.jump_run = []
        if self.mode != TRACK:
            return
        self.miss_time += max(float(dt), 0.0)
        if self.window is not None:
            self.window = min(self.window * WIDEN_PER_MISS,
                              MAX_WINDOW_FRAC * min(fw, fh))
        if self.miss_time >= self.lost_time:
            self.reset()                             # back to full-frame search


# ---------------------------------------------------------------------------
# cursor state machine
# ---------------------------------------------------------------------------
class CursorController:
    """Rate-limited follow + hold-on-loss.

    There is NO acquisition step.  The hover-the-cursor-over-a-square dwell
    mechanic was removed: a position only ever reaches this class after the
    detector and the verifier accepted it, so the first accepted position *is*
    the lock.  The cursor still cannot jump, because the speed cap is applied
    from the very first frame -- acquisition is instant, the movement is not.

    All public positions are in SCREEN pixels.  The speed constant is expressed
    in screen widths so it is resolution independent.
    """

    def __init__(self, screen_w, screen_h, max_speed=MAX_SPEED,
                 loss_timeout=LOSS_TIMEOUT, start=None):
        self.screen_w = float(screen_w)
        self.screen_h = float(screen_h)
        self.loss_timeout = float(loss_timeout)
        self.max_speed_px = float(max_speed) * self.screen_w

        self.cursor = np.array(start if start is not None
                               else (self.screen_w / 2.0, self.screen_h / 2.0),
                               dtype=float)
        self.state = SEARCHING
        self.loss_accum = 0.0

    @property
    def loss_progress(self):
        return min(1.0, self.loss_accum / self.loss_timeout) if self.loss_timeout else 1.0

    def update(self, target_px, dt):
        dt = max(float(dt), 0.0)
        if target_px is not None:
            # A verified position: lock immediately and start following.
            self.state = LOCKED
            self.loss_accum = 0.0
            self.cursor = self._step_toward(self.cursor, target_px, dt)
        elif self.state == LOCKED:
            self.loss_accum += dt              # hold exactly still
            if self.loss_accum >= self.loss_timeout:
                self.state = SEARCHING
                self.loss_accum = 0.0
        return self.cursor.copy(), self.state

    def _step_toward(self, cur, target, dt):
        delta = np.asarray(target, float) - cur
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            return cur.copy()
        step = min(dist, self.max_speed_px * dt)
        return cur + (delta / dist) * step


def frame_to_screen(p, fw, fh, sw, sh):
    return np.array([p[0] / fw * sw, p[1] / fh * sh], dtype=float)


def screen_to_frame(p, fw, fh, sw, sh):
    return np.array([p[0] / sw * fw, p[1] / sh * fh], dtype=float)


def resolve_model_path(name):
    """Resolve a model filename the same way for every flag.

    A bare filename is tried AS GIVEN first (relative to the process cwd), then
    against the bundled model directory, then against this script's directory.
    Without the fallbacks, a double-click -- whose cwd is the exe's own folder --
    cannot find a model even though it is sitting in the bundle next to the exe.
    That silently disabled the verifier in HandCursorDark_v1 and then crashed the
    app on the first detection; see DARK_VARIANT.md.
    """
    if not name:
        return None
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for cand in (name, os.path.join(MODEL_DIR, name), os.path.join(HERE, name)):
        if os.path.isfile(cand):
            return cand
    return None


def load_verifier(path=None):
    """Load the crop verifier, or return None if it is not available.

    Search order: the explicit --verifier path (resolved via resolve_model_path),
    then the model directory, then this script's directory.  A missing model is
    NOT fatal by itself: the locator falls back to the box size gate as in v2 --
    but main() treats an EXPLICIT --verifier that cannot be resolved as fatal,
    because silently degrading is what hid the crash.
    """
    if path is None:
        path = resolve_model_path(VERIFIER_FILENAME)
    else:
        path = resolve_model_path(path)
    if path is None:
        return None, None
    if make_verifier is None:                            # pragma: no cover
        print("WARNING: crop_verifier module unavailable -- "
              "falling back to the box size gate.")
        return None, None
    try:
        return make_verifier(path), path
    except Exception as exc:                             # pragma: no cover
        print(f"WARNING: could not load verifier ({exc}) -- "
              f"falling back to the box size gate.")
        return None, None


def load_blob_verifier(path=None):
    """Load the blob-pipeline-specific verifier, or return None.

    Blob crops are a different distribution from detector crops, so this model is
    trained on crops built through the blob pipeline itself (blob_verifier.py).
    Missing is not fatal: blobs then fall back to the main verifier with the
    8-way rotation search.
    """
    # Route through resolve_model_path for BOTH the default and an explicit
    # --blob-verifier.  A bare explicit name used to be handed to torch.load
    # untouched, so it was resolved against the process cwd only -- the same hole
    # that disabled the main verifier in HandCursorDark_v1.
    path = resolve_model_path(path or BLOB_VERIFIER_FILENAME)
    if path is None or make_verifier is None:
        return None, None
    try:
        return make_verifier(path), path
    except Exception as exc:                             # pragma: no cover
        print(f"WARNING: could not load the blob verifier ({exc}) -- blobs will "
              f"use the main verifier.")
        return None, None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mouse", action="store_true",
                    help="debug window only; do not move the real cursor")
    ap.add_argument("--detector", default=None,
                    help="palm-detector weights (default: palmdetector.pth). Used "
                         "by the dark-specialised variant to load fine-tuned "
                         "weights without touching the originals.")
    ap.add_argument("--verifier", default=None,
                    help=f"crop verifier weights (default: {VERIFIER_FILENAME})")
    ap.add_argument("--verifier-threshold", type=float,
                    default=VERIFIER_THRESHOLD,
                    help="P(hand) floor for a TRACK candidate")
    ap.add_argument("--no-verifier", action="store_true",
                    help="disable the verifier; use the v2 box size gate")
    ap.add_argument("--no-motion-gate", action="store_true",
                    help="disable the acquisition motion gate (pre-v5 "
                         "behaviour: acquire on any verified detection)")
    ap.add_argument("--motion-blobs", action="store_true",
                    help="ALSO propose candidates from motion blobs, outside the "
                         "detector's box, and verifier-check them (v9). Default "
                         "OFF: measured to admit far more moving clutter than "
                         "hands in the target lighting -- see "
                         "MOTION_BLOBS_DESIGN.md")
    ap.add_argument("--acquire-consensus", type=int, default=0, metavar="N",
                    help="require N consecutive REACQUIRE passes that also AGREE "
                         "SPATIALLY (all within --acquire-consensus-tol of the "
                         "run's mean) before committing to a lock. 0 = off (the "
                         "shipped REACQUIRE_HITS=2 behaviour). Used by the dark "
                         "variant; see DARK_CONSENSUS.md")
    ap.add_argument("--acquire-consensus-tol", type=float,
                    default=ACQUIRE_CONSENSUS_TOL, metavar="FRAC",
                    help="agreement radius as a fraction of frame width "
                         f"(default {ACQUIRE_CONSENSUS_TOL})")
    ap.add_argument("--track-consensus", type=int, default=0, metavar="N",
                    help="in TRACK, an in-window candidate that jumps more than "
                         "--track-consensus-tol away from the committed position "
                         "is HELD (the cursor does not move) until N consecutive "
                         "candidates agree on the new place. 0 = off (publish "
                         "every accepted hit, as v10 does)")
    ap.add_argument("--track-consensus-tol", type=float,
                    default=TRACK_CONSENSUS_TOL, metavar="FRAC",
                    help="how far a TRACK candidate may jump from the committed "
                         f"position before it needs confirming (default "
                         f"{TRACK_CONSENSUS_TOL} frame widths)")
    ap.add_argument("--no-motion-blobs", action="store_true",
                    help="force motion-blob proposals OFF even if a wrapper script "
                         "(hand_mouse_dark.py) turns them on. Without this there "
                         "was no way to run the dark build with its fine-tuned "
                         "detector but without the blob path.")
    ap.add_argument("--blob-verifier", default=None,
                    help=f"blob-pipeline-specific verifier (default: "
                         f"{BLOB_VERIFIER_FILENAME} if present); blobs fall back "
                         f"to the main verifier with an 8-way rotation search")
    ap.add_argument("--no-blob-verifier", action="store_true",
                    help="ignore the blob verifier and score blobs with the main "
                         "verifier (pre-v9 behaviour)")
    args = ap.parse_args()
    if args.no_motion_blobs and args.motion_blobs:
        print("NOTE: --no-motion-blobs wins over --motion-blobs.")
        args.motion_blobs = False

    print("Loading detector...")
    if args.detector:
        det_path = resolve_model_path(args.detector)
        if det_path is None:
            raise SystemExit(
                f"ERROR: --detector {args.detector!r} could not be found.\n"
                f"  looked in: the working directory, {MODEL_DIR}, {HERE}")
    else:
        det_path = os.path.join(MODEL_DIR, "palmdetector.pth")
    if not os.path.isfile(det_path):
        raise SystemExit(f"detector weights not found: {det_path}")
    detector = PalmDetector()
    detector.load_weights(det_path)
    if args.detector:
        print(f"Detector weights: {det_path}")
    detector.load_anchors(os.path.join(MODEL_DIR, "anchors.npy"))
    detector.eval()

    verifier = verifier_path = None
    if not args.no_verifier:
        verifier, verifier_path = load_verifier(args.verifier)
    if verifier is not None:
        print(f"Verifier loaded: {verifier_path} "
              f"(threshold {args.verifier_threshold:.2f})")
    elif args.verifier:
        # An EXPLICITLY requested model that cannot be found is a configuration
        # error, not something to degrade past silently.  HandCursorDark_v1 shipped
        # with exactly this hole: started from its own folder, --verifier could not
        # be resolved, the verifier went inactive without complaint, and the app
        # then crashed on the first detection.
        raise SystemExit(
            f"ERROR: --verifier {args.verifier!r} could not be found.\n"
            f"  looked in: the working directory, {MODEL_DIR}, {HERE}\n"
            f"  pass a full path, or omit --verifier to use the default "
            f"({VERIFIER_FILENAME}).")
    else:
        print(f"Verifier NOT active -- TRACK uses the {MIN_BOX_FRAC_TRACK} "
              f"box-size gate (v2 behaviour).")
    print("Models loaded.")

    if pyautogui is None:
        print("NOTE: pyautogui unavailable -- running without moving the cursor.")
        args.no_mouse = True
    else:
        pyautogui.FAILSAFE = True

    if args.no_mouse:
        screen_w, screen_h = 1920, 1080
        print("--no-mouse: using a nominal 1920x1080 cursor space.")
    else:
        screen_w, screen_h = pyautogui.size()

    # NOTE: this must be assigned on EVERY path.  An earlier version only set it
    # inside `if args.motion_blobs:`, so running the app without that flag raised
    # UnboundLocalError at startup -- i.e. the default configuration never worked.
    blob_verifier = blob_path = None
    if args.motion_blobs and not args.no_blob_verifier:
        blob_verifier, blob_path = load_blob_verifier(args.blob_verifier)
        if blob_verifier is None and args.blob_verifier:
            raise SystemExit(
                f"ERROR: --blob-verifier {args.blob_verifier!r} could not be found.")
    if args.motion_blobs and verifier is None and blob_verifier is None:
        # Unconfirmed motion proposals are the measured-unsafe mode (88-94% of
        # OFF-hand blobs accepted).  Refuse the feature rather than run it blind.
        print("WARNING: no verifier available, so motion-blob proposals cannot be "
              "confirmed.")
        print("         Disabling --motion-blobs for this run (unconfirmed blobs "
              "are the unsafe mode).")
        args.motion_blobs = False
    controller = CursorController(screen_w, screen_h)
    motion_gate = None if args.no_motion_gate else MotionGate()
    if args.motion_blobs and motion_gate is None:
        print("NOTE: --motion-blobs needs the motion gate; enabling it.")
        motion_gate = MotionGate()
    locator = HandLocator(detector, verifier=verifier,
                          verifier_threshold=args.verifier_threshold,
                          motion_gate=motion_gate,
                          use_motion_blobs=args.motion_blobs,
                          blob_verifier=blob_verifier,
                          acquire_consensus=args.acquire_consensus,
                          acquire_consensus_tol=args.acquire_consensus_tol,
                          track_consensus=args.track_consensus,
                          track_consensus_tol=args.track_consensus_tol)
    if args.motion_blobs:
        if blob_verifier is not None:
            print(f"Blob verifier loaded: {blob_path} "
                  f"(threshold {locator.blob_threshold:.2f}, "
                  f"{locator.blob_rotations}-way search)")
        else:
            print("Blob verifier NOT active -- blobs use the main verifier at "
                  f"{locator.blob_threshold:.2f} with the 8-way rotation search.")
    smoother = PositionSmoother()
    print(f"screen {screen_w}x{screen_h} | cursor cap {controller.max_speed_px:.0f} px/s "
          f"| loss timeout {LOSS_TIMEOUT:.1f}s (no acquisition dwell)")
    print(f"window = {SEARCH_MARGIN}x box (widen x{WIDEN_PER_MISS}/miss, "
          f"shrink x{SHRINK_PER_HIT}/hit) | smoothing tau {SMOOTHING_TAU}s | "
          f"plausibility {MAX_HAND_SPEED} frame-widths/s")
    if verifier is not None:
        track_gate = f"verifier p_hand >= {args.verifier_threshold:.2f}"
    else:
        track_gate = f"box width >= {MIN_BOX_FRAC_TRACK} x frame"
    print(f"TRACK gate = {track_gate}")
    if locator.acquire_consensus > 1:
        tol_px = locator.acquire_consensus_tol * 640
        print(f"ACQUIRE consensus = ON: {locator.acquire_consensus} consecutive "
              f"passes, each within {locator.acquire_consensus_tol} x frame width "
              f"of the run's mean (~{tol_px:.0f}px at 640 wide). Nothing is "
              f"published (the cursor does not move) until a pass is confirmed.")
    else:
        print(f"ACQUIRE consensus = OFF (commit after {REACQUIRE_HITS} consecutive "
              f"detections, wherever they are)")
    if locator.track_consensus > 1:
        tol_px = locator.track_consensus_tol * 640
        print(f"TRACK consensus = ON: a candidate more than "
              f"{locator.track_consensus_tol} x frame width (~{tol_px:.0f}px at "
              f"640) from the committed position is HELD -- the cursor keeps still "
              f"until {locator.track_consensus} consecutive candidates agree on "
              f"the new place, or {locator.track_consensus_max_hold} frames pass, "
              f"whichever comes first.")
    else:
        print("TRACK consensus = OFF (every accepted detection moves the cursor)")
    if motion_gate is not None:
        print(f"ACQUIRE motion gate = frac >= max({MOTION_MIN}, "
              f"{MOTION_BG_RATIO} x background) on a {MOTION_WIDTH}px-wide "
              f"{MOTION_RING}-frame ring, suspended above "
              f"{GLOBAL_CHANGE_MAX} luma global change")
        print(f"ACQUIRE size gate = box width >= "
              f"{locator.reacquire_min_box_frac} x frame "
              f"(relaxed from {MIN_BOX_FRAC_REACQUIRE_NO_MOTION} in v6; the "
              f"motion gate carries the clutter rejection)")
    else:
        print(f"ACQUIRE motion gate = OFF (--no-motion-gate); size gate back to "
              f"{locator.reacquire_min_box_frac} x frame")
    if args.motion_blobs:
        print(f"ACQUIRE motion blobs = ON: up to {MOTION_BLOB_MAX} blobs/frame, "
              f"ROI x{MOTION_BLOB_ROI_SCALE}, "
              f"{locator.blob_rotations}-way rotation search, verifier-confirmed "
              f"at {args.verifier_threshold:.2f}; verifier ALSO gates REACQUIRE")
        print(f"ACQUIRE blob sensitivity = pixel thr {MOTION_BLOB_PIXEL_THR}, "
              f"area floor {MOTION_BLOB_MIN_SIDE_FRAC} x frame width "
              f"(gate metric unchanged at {MOTION_PIXEL_THR})")
        if blob_verifier is None:
            print("   WARNING: no blob verifier -- measured in the target lighting,"
                  " the main")
            print("   verifier accepts 88-94% of OFF-hand motion blobs. See "
                  "MOTION_BLOBS_DESIGN.md.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index 0).")
        print("Check that a camera is connected and not in use by another app.")
        return

    prev = 0.0
    t0 = time.time()
    raw_pos = None
    smooth_pos = None
    print("Running cursor control... show your hand to the camera to take control.")
    print("Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        now = time.time()
        dt = (now - prev) if prev > 0 else 1.0 / 30.0
        prev = now
        dt = min(max(dt, 1e-3), 0.5)

        raw_pos, info = locator.update(frame, now - t0, dt)      # frame px
        target_screen = None
        if raw_pos is not None:
            target_screen = smoother.update(frame_to_screen(raw_pos, fw, fh,
                                                            screen_w, screen_h), dt)
        else:
            smoother.update(None, dt)

        cursor, state = controller.update(target_screen, dt)
        if not args.no_mouse:
            pyautogui.moveTo(int(round(cursor[0])), int(round(cursor[1])))

        # ---- debug overlay -------------------------------------------------
        if info["window"] and info["mode"] == TRACK and locator.pos is not None:
            half = info["window"] / 2.0
            cv2.rectangle(frame,
                          (int(locator.pos[0] - half), int(locator.pos[1] - half)),
                          (int(locator.pos[0] + half), int(locator.pos[1] + half)),
                          (255, 120, 0), 1)
        if raw_pos is not None:
            cv2.circle(frame, (int(raw_pos[0]), int(raw_pos[1])), 4, (0, 165, 255), -1)
        if target_screen is not None:
            sf = screen_to_frame(target_screen, fw, fh, screen_w, screen_h)
            cv2.circle(frame, (int(sf[0]), int(sf[1])), 3, (255, 0, 255), -1)
        cp = screen_to_frame(cursor, fw, fh, screen_w, screen_h)
        cx, cy = int(cp[0]), int(cp[1])
        # Crosshair only: the old acquisition square is gone, because there is
        # no dwell/hover step to aim at any more.
        cv2.line(frame, (cx - 26, cy), (cx + 26, cy), (255, 255, 0), 1)
        cv2.line(frame, (cx, cy - 26), (cx, cy + 26), (255, 255, 0), 1)

        cv2.putText(frame, f"{info['mode']}  cursor:{state}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if state == LOCKED else (0, 200, 255), 2)
        win_txt = f"{info['window']:.0f}px" if info["window"] else "-"
        cv2.putText(frame, f"window {win_txt}  misses {info['misses']}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if info["verifier"]:
            # p_hand of the accepted candidate; "rej N" counts candidates the
            # classifier threw away this frame (so "no detection" and "detector
            # fired but the verifier said clutter" are distinguishable live).
            p_txt = f"{info['p_hand']:.2f}" if info["p_hand"] is not None else " -  "
            rej = info["n_rejected"]
            cv2.putText(frame, f"p_hand {p_txt}  rej {rej}",
                        (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 255, 120) if rej == 0 else (0, 180, 255), 1)
        cv2.putText(frame, f"loss {controller.loss_progress*100:3.0f}%  "
                           f"FPS {1.0/dt:.1f}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if info["motion_gate"]:
            # Acquisition is the only place motion is required, so this is the
            # "why didn't it take control?" readout: the best motion score seen
            # among this frame's candidates, the threshold it had to beat, and
            # how many candidates the gate killed.
            mf = info["motion_frac"]
            mf_txt = f"{mf:.3f}" if mf is not None else "  -  "
            thr = info["motion_threshold"]
            thr_txt = f"{thr:.3f}" if thr is not None else "  -  "
            colour = (120, 255, 120) if not info["n_motion_rejected"] else (0, 180, 255)
            if motion_gate is not None and motion_gate.suspended():
                mf_txt, colour = "SUSP", (0, 0, 255)
            cv2.putText(frame, f"motion {mf_txt} / {thr_txt}  killed "
                               f"{info['n_motion_rejected']}",
                        (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cv2.imshow("Hand Cursor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
