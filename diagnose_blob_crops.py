"""Step 2.1: why can't the verifier discriminate blob crops?

Same frame, same hand, two preprocessing pipelines:

  training-style crop : anchored on MediaPipe's landmarks, rotation from
                        wrist->MCP, the recipe the verifier was trained on
  blob-pipeline crop  : anchored on the motion blob's centroid, ROI = blob side
                        x 1.2, rotation = best of an 8-way search

If these differ for the SAME hand, "the verifier is out of its trained domain on
blob input" is a preprocessing answer, separate from the low-light domain gap.
This measures both crops' P(hand) and simple image statistics, and separates the
max-over-rotations selection effect from the crop geometry itself.

Usage:
    .venv-blazepalm\\Scripts\\python.exe diagnose_blob_crops.py
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
from hand_mouse_cursor import (MotionGate, hand_roi, rotated_rect_to_points,  # noqa: E402
                               warp_rect, best_rotation_score)
from crop_verifier import make_verifier                     # noqa: E402

CLIP = "VeryLowLight.mp4"
ROI_SCALE = hmc.MOTION_BLOB_ROI_SCALE


def crop_stats(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = float(np.median(g))
    return dict(mean=float(g.mean()), std=float(g.std()),
                contrast=float(g.std() / (g.mean() + 1e-6)),
                frac_far_from_median=float((np.abs(g - med) > 20).mean()))


def mp_hand_crop(frame, lm, fw, fh):
    """The pipeline's own recipe, anchored on MediaPipe's landmarks."""
    L = lm * np.array([fw, fh], dtype=np.float32)
    kp0, kp2 = (float(L[0, 0]), float(L[0, 1])), (float(L[9, 0]), float(L[9, 1]))
    chest = L[[0, 5, 9, 13, 17]].mean(axis=0)
    pw = float(np.hypot(kp2[0] - kp0[0], kp2[1] - kp0[1]))
    if pw < 4:
        return None, None
    box = (chest[0] - 1.3 * pw, chest[1] - 1.3 * pw,
           chest[0] + 1.3 * pw, chest[1] + 1.3 * pw)
    cx, cy, side, rot = hand_roi(box, kp0, kp2)
    crop, _ = warp_rect(frame, rotated_rect_to_points(cx, cy, side, side, rot), 256)
    return crop, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="verifier_cnn.pt")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()
    verifier = make_verifier(os.path.join(HERE, args.verifier))

    z = np.load(os.path.join(HERE, "gt_verylowlight.npz"))
    gt_s, gt_t = z["static_conf0.3"], z["conf0.3"]
    lm_all = z["static_conf0.3_landmarks"]

    def hands(n, window=5):
        out = []
        for gt in (gt_s, gt_t):
            row = gt[gt[:, 0] == n]
            if len(row) and row[0, 1]:
                out.append((float(row[0, 2]), float(row[0, 3]), float(row[0, 4])))
                continue
            present = gt[gt[:, 1] == 1][:, 0]
            if len(present):
                near = present[np.abs(present - n) <= window]
                if len(near):
                    k = near[np.argmin(np.abs(near - n))]
                    row = gt[gt[:, 0] == k]
                    out.append((float(row[0, 2]), float(row[0, 3]), float(row[0, 4])))
        return out

    cap = cv2.VideoCapture(os.path.join(HERE, CLIP))
    gate = MotionGate()
    on_blob, off_blob, train_crop = [], [], []
    on_blob0, off_blob0 = [], []          # single orientation, to isolate the max
    stats_on, stats_train = [], []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if n % 3:                          # every 3rd frame is plenty here
            gate.push(frame)
            continue
        fh, fw = frame.shape[:2]
        gate.push(frame)
        hs = hands(n)
        blobs = gate.blobs(frame.shape)
        for cx, cy, side, _af in blobs:
            r = side * ROI_SCALE
            p_best, _rot = best_rotation_score(frame, cx, cy, r, verifier)
            crop0, _ = warp_rect(frame,
                                 rotated_rect_to_points(cx, cy, r, r, 0.0), 256)
            p0 = float(verifier(crop0))
            near = min((float(np.hypot(cx - h[0], cy - h[1])) / fw for h in hs),
                       default=None)
            if near is None or near > 0.20:
                off_blob.append(p_best)
                off_blob0.append(p0)
            elif near <= 0.12:
                on_blob.append(p_best)
                on_blob0.append(p0)
                stats_on.append(crop_stats(crop0))
                if not np.isnan(lm_all[n - 1]).any():
                    c, sd = mp_hand_crop(frame, lm_all[n - 1], fw, fh)
                    if c is not None:
                        train_crop.append(float(verifier(c)))
                        stats_train.append(crop_stats(c))
    cap.release()

    def summ(name, v):
        if not v:
            print(f"   {name:<46} (none)")
            return
        v = np.array(v)
        print(f"   {name:<46} n={len(v):>4}  median {np.median(v):.3f}  "
              f"p90 {np.percentile(v, 90):.3f}  >= {args.threshold}: "
              f"{100*np.mean(v >= args.threshold):5.1f}%")

    print(f"== P(hand) by crop pipeline, {CLIP} ==")
    summ("blob-pipeline crop, ON the hand (max of 8)", on_blob)
    summ("blob-pipeline crop, ON the hand (single rot 0)", on_blob0)
    summ("TRAINING-style crop, same hands (wrist->MCP)", train_crop)
    summ("blob-pipeline crop, OFF the hand (max of 8)", off_blob)
    summ("blob-pipeline crop, OFF the hand (single rot 0)", off_blob0)

    if on_blob and train_crop:
        a, b = np.median(on_blob), np.median(train_crop)
        print(f"\n   blob/training median ratio for the same hands: "
              f"{a/max(b,1e-6):.2f}x")
    if on_blob and off_blob:
        print(f"   on-hand vs off-hand separation (blob pipeline): "
              f"{np.median(on_blob) - np.median(off_blob):+.3f}")
    if train_crop and on_blob0:
        print(f"   max-over-8 inflation: on-hand "
              f"{np.median(on_blob) - np.median(on_blob0):+.3f}, off-hand "
              f"{np.median(off_blob) - np.median(off_blob0):+.3f}")

    def stat_line(name, rows):
        if not rows:
            print(f"   {name:<28} (none)")
            return
        keys = ("mean", "std", "contrast", "frac_far_from_median")
        vals = {k: float(np.median([r[k] for r in rows])) for k in keys}
        print(f"   {name:<28}{vals['mean']:>9.1f}{vals['std']:>8.1f}"
              f"{vals['contrast']:>10.2f}{vals['frac_far_from_median']:>22.3f}")

    print(f"\n== crop image statistics (medians) ==")
    print(f"   {'crop type':<28}{'mean':>9}{'std':>8}{'contrast':>10}"
          f"{'frac >20 from median':>22}")
    stat_line("blob-pipeline (on hand)", stats_on)
    stat_line("training-style (same hands)", stats_train)


if __name__ == "__main__":
    main()
