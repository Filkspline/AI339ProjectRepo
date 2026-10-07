"""Step 2: build the VeryLowLight training set.

Informed by the Step 1 diagnosis (see build/diag_vll_static.txt):

  * the CNN verifier rejects 79.6% of the genuine hand crops that reach it
    (median P(hand)=0.128 on boxes whose content is the hand), so verifier
    training data for this lighting IS needed;
  * the palm detector fires at a normal rate but its boxes rarely contain the
    hand (35.7% of MediaPipe-hand frames), which verifier data cannot fix;
  * the motion gate costs only 5.8% and the sensor noise here is sigma ~ 1.05,
    far from the ~8 where the gate's margin collapses, so the gate is left alone.

Two data sources:

  1. REAL crops from VeryLowLight.mp4 itself (the target condition):
       positives  - anchored on MediaPipe's own landmarks with the pipeline's
                    rotation recipe (the only trustworthy hand locations here,
                    since the landmark presence head is blind in this light);
       negatives  - our detector's boxes in frames where neither MediaPipe mode
                    sees a hand anywhere near them (clutter in the target light).
     Split by TIME, not randomly: adjacent frames at 15 fps are nearly identical,
     so a random split would leak the test set into training.

  2. SYNTHETIC augmentation of the existing HaGRID + own-room set.  The
     photometric transform is measured, not guessed: per-channel mean/std of the
     target clip vs the source crop (mean+std matching per channel reproduces the
     darkening, the colour cast AND the contrast loss in one step).  Severity is
     sampled, and the target statistics are jittered per sample, so the model
     cannot overfit one exact colour temperature.  Applied to BOTH classes, and
     combined with a random downscale-upscale to simulate range.

Usage:
    .venv-blazepalm\\Scripts\\python.exe build_vll_dataset.py
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (letterbox, box_to_pixels, hand_roi,  # noqa: E402
                               rotated_rect_to_points, warp_rect)

CLIP = "VeryLowLight.mp4"
SAVE_SIZE = 128
CROP_NATIVE = 256
BLOCK = 50                 # frames per split block
HOLDOUT_EVERY = 3          # every 3rd block is held out (~1/3 of the clip)
MIN_BOX = 0.10             # the v6 acquisition size gate


def target_stats(clip, stride=5):
    """Per-channel mean/std of the target clip, measured (float64 accumulation).

    This clip's lighting is not constant -- frame luma climbs from ~36 to ~60
    part way through -- so the statistics are taken over the whole clip, and the
    split below is block-interleaved so both halves span both brightness regimes.
    """
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    s = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    cnt = 0
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        if n % stride:
            continue
        px = f.reshape(-1, 3).astype(np.float64)
        s += px.sum(axis=0)
        s2 += (px ** 2).sum(axis=0)
        cnt += px.shape[0]
    cap.release()
    mean = s / cnt
    var = np.maximum(s2 / cnt - mean ** 2, 0.0)
    return mean, np.sqrt(var), n


def is_holdout(frame_idx):
    return (frame_idx // BLOCK) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1


def photometric(img, target_mean, target_std, severity, rng):
    """Match a crop's per-channel mean/std part-way to the target's."""
    src = img.astype(np.float32)
    out = np.empty_like(src)
    for c in range(3):
        s_mean = float(src[:, :, c].mean())
        s_std = float(src[:, :, c].std()) + 1e-3
        # jitter the target so we do not lock onto one exact colour temperature
        t_mean = target_mean[c] * (1.0 + rng.uniform(-0.15, 0.15))
        t_std = target_std[c] * (1.0 + rng.uniform(-0.15, 0.15))
        matched = (src[:, :, c] - s_mean) * (t_std / s_std) + t_mean
        out[:, :, c] = (1.0 - severity) * src[:, :, c] + severity * matched
    return np.clip(out, 0, 255).astype(np.uint8)


def downscale(img, lo, hi, rng):
    k = rng.uniform(lo, hi)
    if k >= 0.99:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(4, int(w * k)), max(4, int(h * k))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def save(dirpath, name, img):
    os.makedirs(dirpath, exist_ok=True)
    return cv2.imwrite(os.path.join(dirpath, name), img)


def crop_dir(out, label):
    return os.path.join(out, "crops", "clutter" if label == "clutter" else "hand")


def crop_dir_rel(label):
    """The same location relative to the dataset root (what goes in the manifest)."""
    return os.path.join("crops", "clutter" if label == "clutter" else "hand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="verifier_dataset")
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--out", default="vll_dataset")
    ap.add_argument("--aug-severity", type=float, nargs="+",
                    default=[0.5, 0.8, 1.0])
    ap.add_argument("--aug-copies", type=int, default=1,
                    help="synthetic copies per existing crop per severity")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_mean, t_std, n_frames = target_stats(args.clip)
    print(f"target statistics from {args.clip} ({n_frames} frames):")
    print(f"  per-channel mean BGR {t_mean.round(1)}  std {t_std.round(1)}")
    z = np.load(os.path.join(HERE, args.gt))
    gt = z["static_conf0.3"]
    mp_lm = z["static_conf0.3_landmarks"]
    gt_track = z["conf0.3"]

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()

    out = os.path.join(HERE, args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    rows = []

    # ---------------- 1. real crops from the target clip ------------------
    cap = cv2.VideoCapture(os.path.join(HERE, args.clip))
    n = 0
    n_pos_mp = n_pos_det = n_neg = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        fh, fw = frame.shape[:2]
        held = 1 if is_holdout(n) else 0
        row = gt[gt[:, 0] == n]
        mp_present = bool(len(row) and row[0, 1])

        # positives: MediaPipe-anchored, pipeline crop recipe
        if mp_present and not np.isnan(mp_lm[n - 1]).any():
            L = mp_lm[n - 1] * np.array([fw, fh], dtype=np.float32)
            kp0, kp2 = (float(L[0, 0]), float(L[0, 1])), (float(L[9, 0]), float(L[9, 1]))
            cx_lm, cy_lm = (float(row[0, 2]), float(row[0, 3]))
            pw = float(np.hypot(kp2[0] - kp0[0], kp2[1] - kp0[1]))
            if pw > 4:
                box = (cx_lm - 1.3 * pw, cy_lm - 1.3 * pw,
                       cx_lm + 1.3 * pw, cy_lm + 1.3 * pw)
                cx, cy, side, rot = hand_roi(box, kp0, kp2)
                crop, _ = warp_rect(frame, rotated_rect_to_points(cx, cy, side, side, rot),
                                    CROP_NATIVE)
                name = f"vll_mp_{n:04d}.png"
                save(crop_dir(out, "hand"), name,
                     cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA))
                rows.append(dict(crop=os.path.join("crops", "hand", name),
                                 label="hand", source="vll_real_mp", held_out=held,
                                 iou="", score="", box_w_frac=round(2.6 * pw / fw, 4),
                                 gesture=""))
                n_pos_mp += 1

        # our detector's detections this frame
        padded, scale, left, top = letterbox(frame, 256)
        with torch.no_grad():
            dets = det.predict_on_image(padded)
        dets = [d for d in (dets[0] if dets else []) if float(d[18]) >= hmc.MIN_SCORE]

        tr = gt_track[gt_track[:, 0] == n]
        track_pt = (float(tr[0, 2]), float(tr[0, 3])) if len(tr) and tr[0, 1] else None

        for d in dets:
            box = box_to_pixels(d[:4].tolist(), scale, left, top)
            if (box[2] - box[0]) / fw < MIN_BOX:
                continue
            kp0 = ((float(d[4]) * 256 - left) / scale, (float(d[5]) * 256 - top) / scale)
            kp2 = ((float(d[8]) * 256 - left) / scale, (float(d[9]) * 256 - top) / scale)
            cx, cy, side, rot = hand_roi(box, kp0, kp2)
            crop, _ = warp_rect(frame, rotated_rect_to_points(cx, cy, side, side, rot),
                                CROP_NATIVE)
            bc = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            contains = False
            if mp_present:
                contains = (box[0] - 0.15 * (box[2] - box[0]) <= row[0, 2]
                            <= box[2] + 0.15 * (box[2] - box[0])
                            and box[1] - 0.15 * (box[3] - box[1]) <= row[0, 3]
                            <= box[3] + 0.15 * (box[3] - box[1]))
            near_track = (track_pt is not None
                          and float(np.hypot(bc[0] - track_pt[0], bc[1] - track_pt[1]))
                          / fw < 0.15)
            if contains or near_track:
                label, src = "hand", "vll_real_det"
            elif not mp_present:
                # no hand anywhere per the lag-free detector, and not near the
                # tracked hand either -> clutter in the target lighting
                label, src = "clutter", "vll_real_clutter"
            else:
                continue          # ambiguous: hand is present but not here
            name = f"vll_{src}_{n:04d}_{int(box[0]):04d}.png"
            save(crop_dir(out, label), name,
                 cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA))
            rows.append(dict(crop=os.path.join(crop_dir_rel(label), name),
                             label=label, source=src, held_out=held, iou="",
                             score=round(float(d[18]), 4),
                             box_w_frac=round((box[2] - box[0]) / fw, 4), gesture=""))
            if label == "hand":
                n_pos_det += 1
            else:
                n_neg += 1
    cap.release()
    print(f"\nreal crops from {args.clip}:")
    print(f"  MediaPipe-anchored positives : {n_pos_mp} "
          f"({n_pos_mp - sum(1 for r in rows if r['source'] == 'vll_real_mp' and r['held_out'])}"
          f" train / {sum(1 for r in rows if r['source'] == 'vll_real_mp' and r['held_out'])} holdout)")
    print(f"  detector-recipe positives    : {n_pos_det}")
    print(f"  clutter negatives            : {n_neg}")

    # ---------------- 2. synthetic augmentation of the base set -----------
    # Only the TRAINING sources get augmented (and copied).  hagrid_valid /
    # hagrid_test / own_nohand are carried across untouched and stay held out,
    # otherwise augmenting them would leak the HaGRID test split into training
    # and make the generalisation check meaningless.
    base_manifest = os.path.join(HERE, args.base, "manifest.csv")
    base = list(csv.DictReader(open(base_manifest)))
    HELD = ("hagrid_valid", "hagrid_test", "own_nohand")
    n_aug = n_copied = 0
    for r in base:
        img = cv2.imread(os.path.join(HERE, args.base, r["crop"]))
        if img is None:
            continue
        label = r["label"]
        held = 1 if r["source"] in HELD else 0
        # the un-augmented original (daylight performance must survive too)
        base_name = f"base_{os.path.basename(r['crop'])}"
        save(crop_dir(out, label), base_name, img)
        rows.append(dict(crop=os.path.join(crop_dir_rel(label), base_name),
                         label=label, source=r["source"], held_out=held,
                         iou=r.get("iou", ""), score=r.get("score", ""),
                         box_w_frac=r.get("box_w_frac", ""),
                         gesture=r.get("gesture", "")))
        n_copied += 1
        if held:
            continue
        for sev in args.aug_severity:
            for k in range(args.aug_copies):
                a = photometric(img, t_mean, t_std,
                                float(np.clip(sev + rng.uniform(-0.1, 0.1), 0.0, 1.0)),
                                rng)
                a = downscale(a, 0.2, 1.0, rng)
                src = f"{r['source']}_vll_aug"
                name = f"aug_{sev:.1f}_{k}_{os.path.basename(r['crop'])}"
                save(crop_dir(out, label), name, a)
                rows.append(dict(crop=os.path.join(crop_dir_rel(label), name),
                                 label=label, source=src, held_out=0, iou="",
                                 score="", box_w_frac=r.get("box_w_frac", ""),
                                 gesture=""))
                n_aug += 1
    print(f"\nbase set: {n_copied} originals carried across "
          f"(held-out ones untouched), {n_aug} synthetic augmented copies "
          f"(severities {args.aug_severity}, downscale 0.2-1.0, both classes)")

    # ---------------- write the manifest --------------------------------
    os.makedirs(out, exist_ok=True)
    man = os.path.join(out, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} crops -> {out}")
    from collections import Counter
    c = Counter((r["label"], r["source"]) for r in rows)
    for (label, src), cnt in sorted(c.items()):
        ho = sum(1 for r in rows if r["label"] == label and r["source"] == src
                 and r["held_out"] == 1)
        print(f"   {label:<8} {src:<26} {cnt:>6}  (held out {ho})")


if __name__ == "__main__":
    main()
