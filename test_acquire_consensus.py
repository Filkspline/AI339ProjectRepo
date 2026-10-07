"""Does multi-pass spatial consensus actually refuse to commit on a flicker?

Two layers, because they answer different questions:

  PART 1 -- mechanism (deterministic, no detector).  Drives the real HandLocator
  decision path (`_on_hit` / `_on_miss`) with scripted positions.  This is where
  the rule itself is pinned down: an alternating flicker between two locations
  must NEVER commit, a sustained location must commit after exactly N passes, a
  miss must break a run, and a steadily moving hand must still be able to commit.

  PART 2 -- integration (real frames, real gates, scripted detections).  Feeds
  CloseLightMoving's real frames through the REAL update() -- real size gate, real
  motion gate, real plausibility gate, real candidate selection -- but with a stub
  detector that emits scripted boxes, so the flicker pattern is exact.  A stub
  that emits the pattern demonstrates that the whole pipeline, not just the
  arithmetic, refuses to lock.

  The stub detector returns normalised boxes in the 256-space the real decoder
  produces, so detect_full's own letterbox/box_to_pixels maths is exercised
  unchanged.

Run: .venv-blazepalm\\Scripts\\python.exe test_acquire_consensus.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from hand_pipeline import letterbox                                   # noqa: E402
from hand_mouse_cursor import (HandLocator, PlausibilityGate,         # noqa: E402
                              MotionGate, REACQUIRE, TRACK,
                              ACQUIRE_CONSENSUS_N, ACQUIRE_CONSENSUS_TOL)

TOL = ACQUIRE_CONSENSUS_TOL
N = ACQUIRE_CONSENSUS_N
FW = 640.0


def locator(consensus=N, tol=TOL, motion_gate=None, gate=None, track=0,
            track_tol=None):
    """A locator with no detector: PART 1 drives its decision path directly."""
    from hand_mouse_cursor import TRACK_CONSENSUS_TOL
    return HandLocator(None, gate=(gate if gate is not None else PlausibilityGate()),
                       motion_gate=motion_gate, acquire_consensus=consensus,
                       acquire_consensus_tol=tol, track_consensus=track,
                       track_consensus_tol=(TRACK_CONSENSUS_TOL if track_tol is None
                                            else track_tol))


def feed(loc, positions, box_w=120.0, t_step=1.0 / 30.0):
    """Push a scripted sequence of (x, y) or None through the decision path.

    Also records, on `loc.published_log`, the positions that were actually
    PUBLISHED -- what the cursor is told.  That is the quantity the "if it is not
    certain, do not move the mouse there" rule is about: a withheld candidate must
    not appear there at all.
    """
    loc.fw = FW
    t = 0.0
    loc.published_log = []
    for p in positions:
        t += t_step
        if p is None:
            loc._on_miss(t_step, FW, 480.0)
            loc.published_log.append(None)
        else:
            ok = loc._on_hit(np.array(p, dtype=float), box_w)
            loc.published_log.append(np.array(p, dtype=float) if ok else None)
    return loc


def n_published(loc):
    return sum(1 for p in loc.published_log if p is not None)


# ---------------------------------------------------------------------------
# PART 1: the rule itself
# ---------------------------------------------------------------------------
def part1():
    res = []
    A = (260.0, 200.0)          # the hand
    B = (260.0 + 0.12 * FW, 200.0)   # "the armpit": 0.12 frame widths away
    near = (260.0 + 0.03 * FW, 200.0)  # a nearby-but-wrong location (inside tol)

    # 1. alternating A/B for a long time: never commits
    loc = feed(locator(), [A if i % 2 == 0 else B for i in range(30)])
    ok = loc.mode == REACQUIRE and loc.reacquire_hits == 0
    res.append(("flicker hand<->other (30 frames): no lock", ok,
                f"mode={loc.mode}, consensus_resets={loc.consensus_resets}"))

    # 2. sustained A: commits, and exactly at the Nth agreeing pass
    loc = feed(locator(), [A] * N)
    ok = loc.mode == TRACK
    res.append((f"sustained hand: locks after exactly {N} passes", ok,
                f"mode={loc.mode}"))

    # 3. N-1 agreeing passes then a miss: no lock
    loc = feed(locator(), [A] * (N - 1) + [None])
    ok = loc.mode == REACQUIRE
    res.append((f"{N - 1} agreeing passes then a miss: no lock", ok,
                f"mode={loc.mode}"))

    # 4. a miss in the middle of an otherwise sustained sequence: no lock
    loc = feed(locator(), [A, A, None, A, A], )
    ok = loc.mode == REACQUIRE
    res.append(("passes must be CONSECUTIVE (gap breaks the run)", ok,
                f"mode={loc.mode}"))

    # 5. nearby-but-wrong flicker (inside the tolerance) DOES commit -- and that
    #    is correct: at 0.03 fw the offset is inside the 0.06 fw radius the
    #    evaluation itself calls "on the hand", so a lock there is not a
    #    false lock, just a slightly off one.
    loc = feed(locator(), [A, near, A, near, A, near])
    res.append(("flicker WITHIN tolerance commits (harmless: inside the "
                "match radius)", loc.mode == TRACK, f"mode={loc.mode}"))

    # 6. a steadily moving hand (0.03 fw/frame) still commits under the mean rule
    hand = [(100.0 + i * 0.03 * FW, 200.0) for i in range(N)]
    loc = feed(locator(), hand)
    res.append(("slowly moving hand (0.03 fw/frame) still locks",
                loc.mode == TRACK, f"mode={loc.mode}"))

    # 7. a fast moving hand (0.10 fw/frame, faster than the tolerance) does not
    loc = feed(locator(), [(100.0 + i * 0.10 * FW, 200.0) for i in range(N)])
    res.append(("fast moving hand (0.10 fw/frame > tol) does NOT lock "
                "(documented cost)", loc.mode == REACQUIRE, f"mode={loc.mode}"))

    # 8. consensus off = old behaviour: the flicker DOES commit (this is the
    #    regression the feature exists to remove, and it proves the test can fail)
    loc = feed(locator(consensus=0), [A, B])
    res.append(("control: with consensus OFF the flicker commits after "
                "REACQUIRE_HITS", loc.mode == TRACK, f"mode={loc.mode}"))

    # 9. NOTHING is published before the lock is confirmed: this is the
    #    "if it is not certain, do not move the mouse there" requirement.  Before
    #    v4 the first accepted pass was published immediately, so the cursor was
    #    dragged to an unconfirmed candidate even though the LOCK waited.
    loc = feed(locator(), [A, A, A])
    res.append(("nothing published before the lock is confirmed",
                loc.published_log[:N - 1] == [None] * (N - 1)
                and loc.published_log[N - 1] is not None,
                f"published={[p is not None for p in loc.published_log]}"))
    loc = feed(locator(consensus=3), [A])
    res.append(("unconfirmed single pass publishes nothing",
                n_published(loc) == 0, f"published={n_published(loc)}"))
    return res


# ---------------------------------------------------------------------------
# PART 1b: the TRACK hold ("do not move the mouse there unless certain")
# ---------------------------------------------------------------------------
def part1b():
    res = []
    FW_ = FW
    A = (300.0, 240.0)                       # a committed track position
    jump = (300.0 + 0.20 * FW_, 240.0)       # 0.20 fw away: the "armpit"
    TK = 3                                    # track_consensus N
    from hand_mouse_cursor import (TRACK_CONSENSUS_TOL as TT,
                                   TRACK_CONSENSUS_MAX_HOLD as MAXH)
    steady = [A] * TK                         # reach a committed lock

    # 1. a one-frame excursion off the hand never moves the cursor
    loc = feed(locator(track=TK, track_tol=TT), steady + [jump, A])
    moved = any(p is not None and abs(p[0] - jump[0]) < 1e-6
                for p in loc.published_log[TK:])
    res.append(("TRACK: 1-frame jump off the hand never moves the cursor",
                not moved, f"jumps_rejected={loc.jumps_rejected}"))

    # 2. a sustained relocation IS followed (after confirmation)
    loc = feed(locator(track=TK, track_tol=TT), steady + [jump] * (TK + 1))
    last = [p for p in loc.published_log if p is not None][-1]
    on_new = last is not None and abs(last[0] - jump[0]) < 1e-6
    res.append((f"TRACK: sustained move to a new place IS followed ({TK} passes)",
                on_new, f"last={None if last is None else (round(float(last[0])),)}"))

    # 3. smooth tracking is NEVER delayed: within-tolerance motion publishes at once.
    #    Only the frames AFTER the lock commits are counted -- the first N-1
    #    REACQUIRE passes are withheld on purpose by the rule above.
    smooth = [A] + [(A[0] + i * 0.4 * TT * FW_, A[1]) for i in range(1, 6)]
    loc = feed(locator(track=TK, track_tol=TT), steady + smooth[1:])
    after_lock = loc.published_log[TK:]
    res.append(("TRACK: normal within-tolerance motion is published every frame",
                all(p is not None for p in after_lock),
                f"published={sum(1 for p in after_lock if p is not None)} "
                f"of {len(after_lock)} post-lock frames"))

    # 4. the bounded hold: a hand moving steadily FASTER than the tolerance must
    #    still be picked up.  Without the bound the claims never agree with each
    #    other and the cursor would freeze until the 3 s loss timeout.
    fast = [(A[0] + i * 0.25 * FW_, A[1]) for i in range(1, 2 * MAXH + 2)]
    loc = feed(locator(track=TK, track_tol=TT), steady + fast)
    tail = [p for p in loc.published_log[TK:] if p is not None]
    res.append(("TRACK: fast sustained move is picked up within the hold limit "
                "(no permanent freeze)", len(tail) > 0,
                f"published {len(tail)} of {len(fast)} held frames"))

    # 5. control: with the hold OFF the 1-frame excursion DOES move the cursor
    loc = feed(locator(track=0), steady + [jump, A])
    moved = any(p is not None and abs(p[0] - jump[0]) < 1e-6
                for p in loc.published_log[TK:])
    res.append(("CONTROL: without the TRACK hold the same jump moves the cursor",
                moved, f"jumps_rejected={loc.jumps_rejected}"))
    return res


# ---------------------------------------------------------------------------
# PART 2: the whole pipeline over real frames, with a scripted detector
# ---------------------------------------------------------------------------
class ScriptedDetector:
    """Emits whatever box it is told to, in the real detector's output format.

    predict_on_image returns [ [d, ...] ] where d[:4] is a normalised
    [xmin,ymin,xmax,ymax] in the 256-space and d[18] is the score, which is what
    detect_full consumes.
    """

    def __init__(self, script):
        self.script = script        # frame index -> (cx, cy, side) in frame px
        self.i = -1

    def predict_on_image(self, padded_img):
        self.i += 1
        spec = self.script.get(self.i)
        if spec is None:
            return []
        cx, cy, side = spec
        padded, scale, left, top = letterbox(
            np.zeros((480, 640, 3), np.uint8), 256)
        # invert box_to_pixels: norm = (px * scale + left) / 256
        def nx(px):
            return (px * scale + left) / 256.0

        x0, y0 = nx(cx - side / 2.0), nx(cy - side / 2.0)
        x1, y1 = nx(cx + side / 2.0), nx(cy + side / 2.0)
        d = np.zeros(19, dtype=np.float32)
        d[0], d[1], d[2], d[3] = x0, y0, x1, y1
        d[18] = 0.9
        return [[d]]


def part2():
    """Real frames, real gates, scripted detections."""
    res = []
    frames = []
    cap = cv2.VideoCapture(os.path.join(HERE, "CloseLightMoving.mp4"))
    for _ in range(30):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 20:
        return [("integration: CloseLightMoving frames available", False,
                 f"only {len(frames)} frames decoded")]

    # place the scripted box over the real moving hand, so the motion gate sees
    # genuine motion in the box it is asked to judge
    h, w = frames[0].shape[:2]
    truth = {}
    z = np.load(os.path.join(HERE, "gt_clips.npz"))
    gt = z["static:CloseLightMoving.mp4"]
    for row in gt:
        if row[1]:
            truth[int(row[0])] = (float(row[2]), float(row[3]))
    side = 0.30 * w
    # The offset is chosen to be REACHABLE by the plausibility gate (which at
    # 30fps allows MAX_HAND_SPEED x fw / 30 = 64px) but BEYOND the consensus
    # tolerance (0.05 x fw = 32px).  Otherwise the gate, not the consensus, would
    # be doing the rejecting and the test would prove nothing.
    off = 0.08 * w

    def spec(i, extra=0.0):
        if (i + 1) not in truth:
            return None
        x, y = truth[i + 1]
        return (x + extra, y, side)

    def run(script, consensus, gate_on):
        det = ScriptedDetector(script)
        loc = HandLocator(det, gate=PlausibilityGate(),
                          motion_gate=(MotionGate() if gate_on else None),
                          acquire_consensus=consensus,
                          acquire_consensus_tol=TOL)
        hits = 0
        for k, fr in enumerate(frames):
            pos, _info = loc.update(fr, k / 30.0, 1.0 / 30.0)
            if pos is not None:
                hits += 1
        return loc, hits

    steady = {i: spec(i) for i in range(len(frames))}
    flick = {i: spec(i, extra=(off if i % 2 else 0.0)) for i in range(len(frames))}

    cases = [
        ("real gates + sustained hand -> LOCKs", steady, N, True,
         lambda loc, hits: loc.mode == TRACK),
        # consensus is the ONLY mechanism left standing here (no motion gate), so
        # on/off isolates it.  Without the off-control the test could pass for the
        # wrong reason -- which it initially did.
        ("no motion gate + flicker + consensus ON -> no lock", flick, N, False,
         lambda loc, hits: loc.mode == REACQUIRE and loc.consensus_resets >= 3),
        ("CONTROL: same flicker, consensus OFF -> locks (test has teeth)",
         flick, 0, False, lambda loc, hits: loc.mode == TRACK),
    ]
    for name, script, consensus, gate_on, expect in cases:
        loc, hits = run(script, consensus, gate_on)
        res.append((f"integration: {name}", expect(loc, hits),
                    f"mode={loc.mode}, hits={hits}, "
                    f"consensus_resets={loc.consensus_resets}"))
    return res


def main():
    print("== PART 1: consensus rule (deterministic) ==")
    res = part1()
    print("\n== PART 1b: TRACK hold -- do not move unless certain ==")
    res += part1b()
    print("\n== PART 2: real frames + scripted detector (integration) ==")
    res += part2()
    print("\n== summary ==")
    for name, ok, detail in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")
    n = sum(1 for _n, ok, _d in res if ok)
    print(f"\n{len(res)} checks: {n} passed, {len(res) - n} failed")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
