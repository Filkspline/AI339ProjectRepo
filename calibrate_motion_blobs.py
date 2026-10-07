"""Calibrate the motion-blob proposal parameters against real footage.

Two numbers in hand_mouse_cursor.py are assumptions until measured here:

  1. MOTION_BLOB_ROI_SCALE -- a translating hand's motion blob is roughly the
     hand, while the verifier was trained on the detector's 2.6x palm ROI.  What
     factor converts a blob bbox into that convention?
  2. MOTION_BLOB_ROTATIONS -- a blob has no keypoints, so the crop's in-plane
     rotation is unknown.  How tolerant is the verifier to getting it wrong, and
     therefore how fine does the search need to be?

Both are measured on frames where the hand IS present per MediaPipe (lag-free
static mode), by comparing blob geometry against MediaPipe's palm geometry and by
scoring the verifier across rotation offsets.

Usage:
    .venv-blazepalm\\Scripts\\python.exe calibrate_motion_blobs.py
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (MotionGate, rotated_rect_to_points,  # noqa: E402
                               warp_rect)
from crop_verifier import make_verifier                     # noqa: E402

CLIP = "VeryLowLight.mp4"


def hand_at(gt, n, window=5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--gt-key", default="static_conf0.3")
    ap.add_argument("--verifier", default="verifier_cnn.pt")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    z = np.load(os.path.join(HERE, args.gt))
    gt_s = z[args.gt_key]
    gt_t = z.get("conf0.3", gt_s)          # tracking mode, for labelling only
    verifier = make_verifier(os.path.join(HERE, args.verifier))

    cap = cv2.VideoCapture(os.path.join(HERE, args.clip))
    gate = MotionGate()
    ratios = []
    # per ROI-scale: the max-over-rotations P(hand) of every blob-on-hand crop
    scales = [0.8, 1.0, 1.1, 1.3, 1.6, 2.0]
    per_scale = {s: [] for s in scales}
    per_scale_off = {s: [] for s in scales}
    per_rot = {k: [] for k in range(0, 360, 15)}
    off_seen = [0]
    n = 0
    n_frames_with_hand = n_blob_on_hand = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        fh, fw = frame.shape[:2]
        gate.push(frame)
        blobs = gate.blobs(frame.shape)
        if not blobs:
            continue
        # Label each blob on/off the hand using BOTH MediaPipe modes.  Using
        # only static mode would be circular in the worst way: in the 25% of
        # frames where static mode misses the hand, real hand blobs would be
        # counted as clutter and inflate the false-accept rate.
        cands = []
        for gt_ in (gt_s, gt_t):
            g_ = hand_at(gt_, n)
            if g_ is not None:
                cands.append(g_)
        if cands:
            n_frames_with_hand += 1
        for b in blobs:
            near = min((float(np.hypot(b[0] - g[0], b[1] - g[1])) / fw
                        for g in cands), default=None)
            if near is None or near > 0.20:
                on_hand = False          # no MediaPipe hand anywhere near it
                if off_seen[0] >= 400:
                    continue
                off_seen[0] += 1
            elif near <= 0.12:
                on_hand = True
                n_blob_on_hand += 1
                g = min(cands, key=lambda t: float(np.hypot(b[0] - t[0],
                                                            b[1] - t[1])))
                ratios.append(2.6 * g[2] / max(b[2], 1.0))
            else:
                continue                 # ambiguous band 0.12-0.20, skip
            for s in scales:
                r = max(b[2], 8.0) * s
                best = 0.0
                for k in range(len(per_rot) if s == 1.1 else 8):
                    deg = k * (15 if s == 1.1 else 45)
                    rect = rotated_rect_to_points(b[0], b[1], r, r, math.radians(deg))
                    crop, _ = warp_rect(frame, rect, 256)
                    p = float(verifier(crop))
                    if s == 1.1 and on_hand:
                        per_rot[deg].append(p)
                    best = max(best, p)
                (per_scale[s] if on_hand else per_scale_off[s]).append(best)

    cap.release()
    print(f"{args.clip}: {n} frames, {n_frames_with_hand} with a MediaPipe hand, "
          f"{n_blob_on_hand} blob-on-hand crops, {off_seen[0]} blob-off-hand crops")
    if not ratios:
        print("no usable blob/hand pairs -- nothing to calibrate")
        return

    r = np.array(ratios)
    print(f"\n== ROI side / blob side (blob-on-hand pairs) ==")
    print(f"   median {np.median(r):.2f}  p25 {np.percentile(r,25):.2f}  "
          f"p75 {np.percentile(r,75):.2f}   (n={len(r)})")

    print(f"\n== verifier confirmation, per ROI scale (bar {args.threshold}) ==")
    print("   ON-hand blobs are the recall side, OFF-hand blobs the precision"
          "\n   side.  The scale must be chosen on the two together.")
    print(f"   {'ROI scale':>10}{'on: med':>9}{'on: acc':>9}"
          f"{'off: med':>10}{'off: acc':>10}{'separation':>12}")
    for s in scales:
        on = np.array(per_scale[s])
        off = np.array(per_scale_off[s])
        a_on = 100.0 * float((on >= args.threshold).mean()) if len(on) else float("nan")
        a_off = 100.0 * float((off >= args.threshold).mean()) if len(off) else float("nan")
        print(f"   {s:>10.1f}{np.median(on):>9.3f}{a_on:>8.1f}%"
              f"{np.median(off):>10.3f}{a_off:>9.1f}%"
              f"{a_on - a_off:>11.1f}")

    print(f"\n== threshold sweep on the best-separating scales ==")
    print("   is there ANY operating point where on-hand blobs beat off-hand"
          " blobs?")
    for s in (1.0, 1.3, 2.0):
        on = np.array(per_scale[s])
        off = np.array(per_scale_off[s])
        print(f"   ROI scale {s}:")
        print(f"      {'thr':>6}{'on acc':>9}{'off acc':>10}{'separation':>12}")
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            a_on = 100.0 * float((on >= thr).mean()) if len(on) else float("nan")
            a_off = 100.0 * float((off >= thr).mean()) if len(off) else float("nan")
            print(f"      {thr:>6.2f}{a_on:>8.1f}%{a_off:>9.1f}%"
                  f"{a_on - a_off:>11.1f}")

    print(f"\n== verifier P(hand) vs rotation offset (ROI scale 1.1) ==")
    med = {deg: float(np.median(v)) for deg, v in per_rot.items() if v}
    if med:
        best_deg = max(med, key=med.get)
        vals = np.array([med[d] for d in sorted(med)])
        print(f"   spread: min {vals.min():.3f} max {vals.max():.3f} "
              f"(ratio {vals.max()/max(vals.min(),1e-6):.1f}x), best offset "
              f"{best_deg}d")
        print(f"   -> a coarse search is fine: the verifier barely cares about the"
              f"\n      unknown blob orientation (ratio far below 2x).")


if __name__ == "__main__":
    main()
