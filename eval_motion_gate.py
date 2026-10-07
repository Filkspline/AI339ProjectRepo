"""Integrated validation of the v5 acquisition motion gate on the real clips.

Runs the ACTUAL pipeline (HandLocator + crop verifier + MotionGate) over the
recorded clips -- not the standalone probe -- and answers the three questions
the design was signed off on:

  A. does the gate change what the pipeline does, and how?
     (acquisitions, hand-matched vs not, hand frames tracked / clutter locks,
     gate ON vs the --no-motion-gate rollback)
  B. does the held-still separation survive integration?
     (CloseLightStill/CloseDarkStill must still acquire; the no-hand clips must
     acquire nothing at all)
  C. does the auto-exposure immunity survive integration?
     (a +-15% gain ramp injected into no-hand clips: how many candidates cross
     the motion floor purely because of AE, and are they stopped?)

Ground truth is MediaPipe's own hands solution (gt_clips.npz from gt_clips.py),
matched with a hand-scaled radius and +-5 frame borrowing, exactly as in
eval_verifier_vs_gt.py.

Usage:
    .venv-blazepalm\\Scripts\\python.exe eval_motion_gate.py
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import HandLocator, PlausibilityGate, MotionGate  # noqa: E402
from crop_verifier import make_verifier                     # noqa: E402

CLIPS_HAND = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
              "FarLight.mp4", "FarDark.mp4"]
CLIPS_NONE = ["NohandLight.mp4", "NohandDark.mp4"]
DT = 1.0 / 30.0
MATCH_R = 0.03          # frame widths, floor; scaled up by palm width at match time
ORIG_DETECT_FULL = hmc.detect_full      # kept so gate overrides never nest


def gt_at(gt, n, window=5):
    """MediaPipe hand position for frame n, borrowing from +-window frames.

    MediaPipe only sees the far hand in ~60% of frames, so a miss is not
    evidence of absence.
    """
    row = gt[gt[:, 0] == n]
    if len(row) and row[0, 1]:
        return float(row[0, 2]), float(row[0, 3]), float(row[0, 4])
    present = gt[gt[:, 1] == 1][:, 0]
    if len(present):
        near = present[np.abs(present - n) <= window]
        if len(near):
            k = near[np.argmin(np.abs(near - n))]
            row = gt[gt[:, 0] == k]
            return float(row[0, 2]), float(row[0, 3]), float(row[0, 4])
    return None


USE_MOTION_BLOBS = [False, None, 0.30]


def replay(det, clip, verifier, motion_gate, gt, max_frames, gain_amp=0.0,
           match_r=None):
    """Run the real locator frame by frame; record acquisitions and outcomes.

    `match_r` is the floor of the hand-scaled match radius, in frame widths.
    The v3/v4 baseline numbers were measured with 0.12 (eval_verifier_vs_gt.py),
    so pass the same value to compare configs; 0.03 is the strict view.
    """
    if match_r is None:
        match_r = MATCH_R
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=verifier,
                      motion_gate=motion_gate,
                      use_motion_blobs=USE_MOTION_BLOBS[0],
                      blob_verifier=USE_MOTION_BLOBS[1],
                      verifier_threshold=USE_MOTION_BLOBS[2])
    n = accept = track = 0
    hit = fp = 0
    acquisitions = []       # (frame, matched_to_hand, distance_frac, frac)
    cand_fracs = []         # every REACQUIRE candidate's motion score
    n_suspended = 0
    n_crossed = 0           # candidates that met the absolute floor
    prev_mode = loc.mode
    t = 0.0
    while True:
        ret, frame = cap.read()
        if not ret or n >= max_frames:
            break
        n += 1
        t += DT
        if gain_amp:
            frame = np.clip(frame.astype(np.float32)
                            * (1.0 + gain_amp * np.sin(2 * np.pi * n / 30.0)),
                            0, 255).astype(np.uint8)
        fw = frame.shape[1]
        pos, info = loc.update(frame, t, DT)

        if info["motion_gate"] and loc.mode == hmc.REACQUIRE:
            if info["motion_frac"] is not None:
                cand_fracs.append(float(info["motion_frac"]))
                if info["motion_frac"] >= hmc.MOTION_MIN:
                    n_crossed += 1
            if motion_gate is not None and motion_gate.suspended():
                n_suspended += 1

        if info["mode"] == hmc.TRACK:
            track += 1
        if info["mode"] == hmc.TRACK and prev_mode == hmc.REACQUIRE:
            g = gt_at(gt, n)
            if g is None:
                acquisitions.append((n, False, float("nan"), info["motion_frac"]))
            else:
                d = float(np.hypot(pos[0] - g[0], pos[1] - g[1])) / fw
                radius = max(match_r, 0.6 * g[2] / fw)
                acquisitions.append((n, d <= radius, d, info["motion_frac"]))
        prev_mode = info["mode"]

        if pos is not None:
            accept += 1
            g = gt_at(gt, n)
            if g is None:
                fp += 1
            else:
                d = float(np.hypot(pos[0] - g[0], pos[1] - g[1])) / fw
                if d <= max(match_r, 0.6 * g[2] / fw):
                    hit += 1
                else:
                    fp += 1
    cap.release()
    return dict(n=n, accept=accept, track=track, hit=hit, fp=fp,
                acquisitions=acquisitions, cand_fracs=cand_fracs,
                n_suspended=n_suspended, n_crossed=n_crossed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--match-r", type=float, default=None,
                    help="floor of the hand-scaled match radius in screen widths; "
                         "0.12 reproduces the v3/v4 baseline matching, 0.03 is strict")
    ap.add_argument("--ae-amp", type=float, default=0.15)
    ap.add_argument("--verifier", default=hmc.VERIFIER_FILENAME)
    ap.add_argument("--threshold", type=float, default=hmc.VERIFIER_THRESHOLD)
    ap.add_argument("--motion-blobs", action="store_true",
                    help="enable motion-blob candidates")
    ap.add_argument("--no-blob-verifier", action="store_true")
    ap.add_argument("--reacquire-gate", type=float, default=None,
                    help="override the full-frame box-size gate used while "
                         "REACQUIREing (default: the shipped 0.15)")
    ap.add_argument("--sweep-acquire-gate", action="store_true",
                    help="run the gate-ON pipeline for several REACQUIRE size "
                         "gates, to see whether the motion gate can pay for a "
                         "looser acquisition size gate (far-range acquisition)")
    args = ap.parse_args()

    # these three globals are read inside replay(), which is what actually builds
    # the locator -- without them --motion-blobs / --threshold / --no-blob-verifier
    # were silently ignored by this harness
    USE_MOTION_BLOBS[0] = args.motion_blobs
    USE_MOTION_BLOBS[2] = args.threshold
    if args.motion_blobs:
        USE_MOTION_BLOBS[1] = (None if args.no_blob_verifier
                               else load_blob_verifier(None)[0])
        print(f"blob verifier: "
              f"{'loaded' if USE_MOTION_BLOBS[1] else 'NONE (main, 8-way)'}")
    print(f"verifier {args.verifier} @ threshold {args.threshold}")

    gt_path = os.path.join(HERE, "gt_clips.npz")
    if not os.path.isfile(gt_path):
        raise SystemExit("gt_clips.npz missing -- run gt_clips.py in the "
                         "mediapipe venv first")
    z = np.load(gt_path)

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    verifier = make_verifier(os.path.join(HERE, args.verifier))

    def set_reacquire_gate(value):
        """Override the full-frame size gate used in REACQUIRE.

        `detect_full` binds MIN_BOX_FRAC_REACQUIRE as a default argument, so the
        module constant cannot simply be reassigned -- wrap the function.  The
        ORIGINAL function is captured once, so repeated calls do not nest
        wrappers.
        """
        if value is None:
            hmc.detect_full = ORIG_DETECT_FULL
            return
        # the locator passes min_box_frac explicitly (v6), so drop any incoming
        # value before applying the override or the call collides
        hmc.detect_full = (lambda det_, frame_, **k:
                           ORIG_DETECT_FULL(
                               det_, frame_, min_box_frac=value,
                               **{kk: vv for kk, vv in k.items()
                                  if kk != "min_box_frac"}))

    if args.sweep_acquire_gate:
        print("== acquisition size gate sweep, motion gate ON ==")
        print("   can the motion gate pay for a looser REACQUIRE size gate?")
        print("   a far hand can NEVER pass the 0.15 gate in REACQUIRE, so before")
        print("   v5 far-range tracking was bootstrapped by a lock on static")
        print("   clutter that happened to be big enough.  The motion gate refuses")
        print("   exactly that, so this asks whether loosening the size gate (now")
        print("   safe, because static small candidates are motion-rejected)")
        print("   restores far-range acquisition honestly.\n")
        clips = ["FarLight.mp4", "FarDark.mp4", "CloseLightStill.mp4",
                 "NohandLight.mp4", "NohandDark.mp4"]
        print(f"   {'gate':<7}" + "".join(f"{c.split('.')[0][:12]:>16}" for c in clips))
        print(f"   {'':<7}" + "".join(f"{'acq onhand hit fp':>16}" for _ in clips))
        for gate in (0.15, 0.12, 0.10, 0.08, 0.06):
            set_reacquire_gate(gate)
            line = f"   {gate:<7.2f}"
            for clip in clips:
                r = replay(det, clip, verifier, MotionGate(), z[clip],
                           args.frames)
                good = sum(1 for _f, m, _d, _x in r["acquisitions"] if m)
                line += (f"{len(r['acquisitions']):>6}{good:>5}"
                         f"{r['hit']:>5}{r['fp']:>5}")
            print(line)
        set_reacquire_gate(None)
        print("\n   acq=acquisitions, onhand=those within the match radius at the")
        print("   acquisition frame, hit/fp=accepted frames on/off the hand.")
        print("   The no-hand clips must stay at 0 acquisitions at EVERY gate.")
        return

    set_reacquire_gate(args.reacquire_gate)
    if args.reacquire_gate is not None:
        print(f"(REACQUIRE size gate overridden to {args.reacquire_gate})\n")

    # ---- A + B: gate ON vs OFF over every clip --------------------------
    print("== A/B. integrated pipeline, gate ON vs the --no-motion-gate rollback ==")
    hdr = (f"{'clip':<21}{'acq OFF':>8}{'acq ON':>7}{'bad acq ON':>11}"
           f"{'trk OFF':>8}{'trk ON':>7}{'hit ON':>7}{'fp ON':>6}"
           f"{'killed':>7}{'susp':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = dict(n=0, a_off=0, a_on=0, bad=0, t_off=0, t_on=0, hit=0, fp=0,
               killed=0)
    for clip in CLIPS_HAND + CLIPS_NONE:
        gt = z[clip]
        off = replay(det, clip, verifier, None, gt, args.frames,
                    match_r=args.match_r)
        on = replay(det, clip, verifier, MotionGate(), gt, args.frames,
                    match_r=args.match_r)
        bad = sum(1 for _f, m, _d, _x in on["acquisitions"] if not m)
        print(f"{clip:<21}{len(off['acquisitions']):>8}"
              f"{len(on['acquisitions']):>7}{bad:>11}"
              f"{100*off['track']/max(off['n'],1):>7.0f}%"
              f"{100*on['track']/max(on['n'],1):>6.0f}%"
              f"{on['hit']:>7}{on['fp']:>6}"
              f"{sum(1 for _f, m, _d, _x in on['acquisitions'] if not m):>7}"
              f"{on['n_suspended']:>6}")
        tot["n"] += on["n"]
        tot["a_off"] += len(off["acquisitions"])
        tot["a_on"] += len(on["acquisitions"])
        tot["bad"] += bad
        tot["t_off"] += off["track"]
        tot["t_on"] += on["track"]
        tot["hit"] += on["hit"]
        tot["fp"] += on["fp"]
    print("-" * len(hdr))
    print(f"{'TOTAL':<21}{tot['a_off']:>8}{tot['a_on']:>7}{tot['bad']:>11}"
          f"{100*tot['t_off']/max(tot['n'],1):>7.0f}%"
          f"{100*tot['t_on']/max(tot['n'],1):>6.0f}%"
          f"{tot['hit']:>7}{tot['fp']:>6}")
    print("\n'acq' = number of times the locator committed to a lock "
          "(SEARCHING->TRACK).\n'bad acq ON' = acquisitions not within the match "
          "radius of the MediaPipe hand.\n'hit'/'fp' = accepted frames on/off the "
          "hand, same matching as the v3 eval.")

    # ---- B, stated plainly ----------------------------------------------
    print("\n== B. held-still separation, integrated ==")
    for clip in CLIPS_HAND + CLIPS_NONE:
        gt = z[clip]
        on = replay(det, clip, verifier, MotionGate(), gt, args.frames,
                    match_r=args.match_r)
        acqs = on["acquisitions"]
        good = sum(1 for _f, m, _d, _x in acqs if m)
        fracs = [x for _f, _m, _d, x in acqs if x is not None]
        f_txt = f"{np.median(fracs):.3f}" if fracs else "  -  "
        print(f"   {clip:<21} acquisitions {len(acqs):>3}  "
              f"on-hand {good:>3}  median motion frac at acquisition {f_txt}")

    # ---- C: AE injection -------------------------------------------------
    print(f"\n== C. auto-exposure immunity, integrated (+-{args.ae_amp:.0%} gain "
          f"ramp injected) ==")
    print("   a no-hand or far clip: every acquisition here is a false positive")
    print(f"   {'clip':<21}{'acq clean':>10}{'acq +AE':>9}"
          f"{'cand>floor clean':>18}{'cand>floor +AE':>16}{'susp +AE':>10}")
    for clip in CLIPS_NONE + ["FarLight.mp4", "FarDark.mp4"]:
        gt = z[clip]
        clean = replay(det, clip, verifier, MotionGate(), gt, args.frames,
                       match_r=args.match_r)
        ae = replay(det, clip, verifier, MotionGate(), gt, args.frames,
                    gain_amp=args.ae_amp, match_r=args.match_r)
        print(f"   {clip:<21}{len(clean['acquisitions']):>10}"
              f"{len(ae['acquisitions']):>9}{clean['n_crossed']:>18}"
              f"{ae['n_crossed']:>16}{ae['n_suspended']:>10}")
    print("\n   'cand>floor' counts REACQUIRE candidates whose motion frac met the"
          "\n   absolute floor; the AE column shows how many that would have been"
          "\n   WITHOUT the gain compensation and the suspension guard.")


if __name__ == "__main__":
    main()
