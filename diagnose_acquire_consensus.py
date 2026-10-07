"""Measure what a multi-pass spatial consensus on REACQUIRE would have to reject.

This does NOT change the app.  It replays the REAL acquisition path -- dark
detector, 130k verifier @0.61, motion blobs, motion gate, plausibility gate -- over
every clip, but pins the locator in REACQUIRE so the full-frame candidate sequence
is recorded frame by frame instead of disappearing into a lock.

What it answers, with real numbers rather than intuition:

  1. When the pipeline accepts a position, how far does it move between
     consecutive accepted frames -- split by whether BOTH ends were on the real
     hand, and by whether it flipped between hand and something else?  That
     distribution is the only defensible basis for a spatial tolerance: it must
     sit above genuine hand motion and below the hand/not-hand flip distance.
  2. For a given (rule, N, tolerance), how many commits land on a false location
     and how much longer the first commit takes.  That second number is the
     latency/recall cost of asking for more passes.

Usage:
    .venv-blazepalm\\Scripts\\python.exe diagnose_acquire_consensus.py
    .venv-blazepalm\\Scripts\\python.exe diagnose_acquire_consensus.py --clips CloseDarkStill.mp4
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                                   # noqa: E402
from crop_verifier import make_verifier                              # noqa: E402
from hand_mouse_cursor import (HandLocator, PlausibilityGate,        # noqa: E402
                               MotionGate, REACQUIRE, load_blob_verifier)
# the yardstick is imported, never re-implemented: identical truth set, identical
# match radius, so these numbers are comparable with evaluate_pipelines.py
from evaluate_pipelines import CLIPS, matched, truth_at              # noqa: E402

DARK_DET = "palmdetector_dark.pth"
DARK_VER = "verifier_cnn_big.pt"
DARK_THR = 0.61
TOLS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075)
NS = (2, 3, 4)


def load_dark():
    det = PalmDetector()
    det.load_weights(os.path.join(HERE, DARK_DET))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    ver = make_verifier(os.path.join(HERE, DARK_VER))
    blob_ver, _p = load_blob_verifier(None)
    return det, ver, blob_ver


def record(det, ver, blob_ver, clip, gt_path, key, max_frames, refresh=False):
    """One (n, accepted_pos_or_None, truth_list, frame_w) row per frame, REACQUIRE pinned.

    Cached to build/consensus_rows_<clip>.npz: running the detector over every clip
    takes minutes, and the simulation below is meant to be re-run cheaply.
    """
    cache = os.path.join(HERE, "build",
                         f"consensus_rows_{clip.replace('.mp4', '')}.npz")
    if os.path.isfile(cache) and not refresh:
        z = np.load(cache, allow_pickle=True)
        rows = []
        for i in range(len(z["n"])):
            hp = z["hands"][i]
            rows.append((int(z["n"][i]),
                         None if np.isnan(z["pos"][i][0])
                         else (float(z["pos"][i][0]), float(z["pos"][i][1])),
                         [tuple(float(x) for x in h) for h in hp],
                         int(z["fw"][i])))
        return rows

    z = np.load(os.path.join(HERE, gt_path))
    gt_static = z[key]
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=ver,
                      verifier_threshold=DARK_THR, motion_gate=MotionGate(),
                      use_motion_blobs=True, blob_verifier=blob_ver)
    rows = []
    n = 0
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok or n >= max_frames:
            break
        n += 1
        now = time.time()
        dt = (now - prev) if prev is not None else 1.0 / 30.0
        prev = now
        pos, _info = loc.update(frame, now, dt or 1.0 / 30.0)
        # Pin REACQUIRE: without this the locator would commit after
        # REACQUIRE_HITS and the acquisition sequence would stop being observable.
        loc.mode = REACQUIRE
        loc.reacquire_hits = 0
        loc.window = None
        rows.append((n, None if pos is None else (float(pos[0]), float(pos[1])),
                     truth_at(gt_static, None, n), frame.shape[1]))
    cap.release()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache,
             n=np.array([r[0] for r in rows]),
             pos=np.array([[np.nan, np.nan] if r[1] is None else list(r[1])
                           for r in rows]),
             fw=np.array([r[3] for r in rows]),
             hands=np.array([np.array(r[2], dtype=float).reshape(-1, 3)
                             for r in rows], dtype=object))
    return rows


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def jump_stats(rows, clip):
    """Consecutive-accepted-frame displacements, split by truth agreement."""
    both_tp, both_fp, flip = [], [], []
    for (n0, p0, h0, fw), (n1, p1, h1, _fw) in zip(rows, rows[1:]):
        if p0 is None or p1 is None or n1 != n0 + 1:
            continue
        d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / fw
        t0 = bool(h0) and matched(p0, h0, fw)
        t1 = bool(h1) and matched(p1, h1, fw)
        if t0 and t1:
            both_tp.append(d)
        elif not t0 and not t1:
            both_fp.append(d)
        else:
            flip.append(d)
    print(f"\n  {clip}: consecutive-accepted-frame displacement (fraction of "
          f"frame width), n={len(rows)} frames")
    for name, v in (("hand -> hand  (genuine motion)", both_tp),
                    ("hand <-> other (a FLIP)", flip),
                    ("other -> other (false on false)", both_fp)):
        if not v:
            print(f"    {name:<34} none")
            continue
        print(f"    {name:<34} n={len(v):<5} median {pct(v, 50):.4f}  "
              f"p75 {pct(v, 75):.4f}  p90 {pct(v, 90):.4f}  "
              f"p95 {pct(v, 95):.4f}  max {max(v):.4f}")
    if both_tp and flip:
        print(f"    -> FLIP median {pct(flip, 50):.4f} vs hand-motion "
              f"p95 {pct(both_tp, 95):.4f}: "
              f"{'SEPARABLE' if pct(flip, 50) > pct(both_tp, 95) else 'OVERLAPPING'}"
              f"  (flip p10 {pct(flip, 10):.4f})")
    return dict(both_tp=both_tp, flip=flip, both_fp=both_fp)


def simulate(rows, n_need, tol, rule="mean"):
    """Replay the recorded sequence through a consensus rule.

    Strictly consecutive accepted hits.  Any miss -- or any hit that fails the
    spatial agreement test -- starts a fresh run.  A run commits at n_need hits.

    rule="mean": the new hit must be within tol of the RUN MEAN.  This is the
        literal reading of "N passes that are all roughly the same place", and it
        is strict about a moving hand: a hand crossing the frame at 0.05 fw/frame
        drifts away from a 2-frame mean quickly.
    rule="last": the new hit must be within tol of the PREVIOUS accepted hit.
        A steadily moving hand chains fine frame to frame; a jump breaks the run.
        Looser overall, since a slow drift can walk tol per frame.
    """
    commits = []
    run = []
    for n, p, hands, fw in rows:
        if p is None:
            run = []
            continue
        pv = np.array(p, dtype=float)
        if run:
            ref = (np.mean([r[1] for r in run], axis=0) if rule == "mean"
                   else run[-1][1])
            if float(np.hypot(pv[0] - ref[0], pv[1] - ref[1])) / fw > tol:
                run = []
        run.append((n, pv))
        if len(run) >= n_need:
            commits.append((n, p, bool(hands) and matched(p, hands, fw)))
            run = []
    return commits


def first_lock(rows, n_need, tol, rule):
    """(delay_frames, on_hand, ever) for the first commit at/after the first truth frame.

    The delay is the honest latency cost: how much longer the user waits before
    ANYTHING is reported, not the per-frame wall clock.
    """
    commits = simulate(rows, n_need, tol, rule)
    first_truth = next((n for n, _p, h, _f in rows if h), None)
    if first_truth is None:
        return None, None, False
    later = [c for c in commits if c[0] >= first_truth]
    if not later:
        return None, None, False
    return later[0][0] - first_truth, later[0][2], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--refresh", action="store_true",
                    help="re-run the detector rather than reusing cached rows")
    args = ap.parse_args()

    det, ver, blob_ver = load_dark()
    print(f"dark config: {DARK_DET} + {DARK_VER} @{DARK_THR} + motion blobs ON")
    print("truth: static MediaPipe per frame (identical yardstick to "
          "evaluate_pipelines.py)")

    wanted = set(args.clips) if args.clips else None
    recorded = {}
    for clip, gtf, key, kind in CLIPS:
        if wanted and clip not in wanted:
            continue
        if not os.path.isfile(os.path.join(HERE, clip)):
            continue
        if not os.path.isfile(os.path.join(HERE, gtf)):
            continue
        rows = record(det, ver, blob_ver, clip, gtf, key, args.frames,
                      refresh=args.refresh)
        recorded[clip] = (rows, kind)
        print(f"\n=== {clip} ({kind}) ===")
        jump_stats(rows, clip)

    for rule in ("mean", "last"):
        print(f"\n\n== consensus simulation, rule = {rule} ==")
        print("baseline = shipped behaviour: REACQUIRE_HITS=2, no spatial test")
        for n_need, tol in [(2, 1e9)] + [(n, t) for n in NS for t in TOLS]:
            label = "baseline" if tol > 1 else f"N={n_need} tol={tol:<6.3f}"
            bits = []
            tot = fals = 0
            late_ok, late_all, first_false = [], [], 0
            for clip, (rows, kind) in recorded.items():
                commits = simulate(rows, n_need, tol, rule)
                tot += len(commits)
                fals += sum(1 for c in commits if not c[2])
                if kind != "hand":
                    continue
                d, on_hand, ever = first_lock(rows, n_need, tol, rule)
                short = clip.split(".")[0].replace("Close", "C").replace("Far", "F")
                if not ever:
                    bits.append(f"{short}:NEVER")
                    continue
                late_all.append(d)
                if on_hand:
                    late_ok.append(d)
                else:
                    first_false += 1
                bits.append(f"{short}:{d}{'' if on_hand else ' WRONG'}")
            fp = 100 * fals / max(tot, 1)
            ma = float(np.mean(late_all)) if late_all else float("nan")
            mk = float(np.mean(late_ok)) if late_ok else float("nan")
            print(f"  {label:<17} commits {tot:<5} false {fals:<4} ({fp:5.1f}%)  "
                  f"lock delay all {ma:5.1f}f correct {mk:5.1f}f, "
                  f"wrong-first-lock {first_false} clip(s)")
            print(f"      {' '.join(bits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
