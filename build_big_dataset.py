"""Part 1: build a much larger verifier training set from real HaGRIDv2 volume.

What was used before: only the 1.18 GB `zhouxzh/portable-hagridv2-mediapipe-hand`
repo (4 shards -> 11,203 crops).  Those shards carry palm boxes and keypoints, so
crops could be mined by geometry.

The `testdummyvt/hagRIDv2_512px` mirror (the real 64 GB 512px set, 1503 train
shards) carries only `image` + `label` (gesture class), so crops are mined
differently: for an image labelled with a gesture (any class except 13 =
no_gesture) the image is known to contain a hand, so the palm detector's top
detection on it is used as a hand crop.  That is image-label-supervised mining --
noisier than box-anchored mining, and stated as such -- and the labels it can
produce are exactly the crop geometry the detector path feeds the verifier.

`no_gesture` images are SKIPPED: per the dataset card they are "natural hand
postures", i.e. they contain hands, so they are neither clean positives nor
clutter.

Negative/clutter crops keep the sources already collected (HaGRID portable
detector FPs, our own room, VeryLowLight clutter), and the established
augmentation discipline is applied to BOTH classes with the same helpers as
build_vll_dataset.py, imported rather than re-implemented.

Usage:
    .venv-blazepalm\\Scripts\\python.exe build_big_dataset.py --max-hagrid 60000
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (letterbox, box_to_pixels, hand_roi,  # noqa: E402
                               rotated_rect_to_points, warp_rect)
from handlandmarks import HandLandmarks                     # noqa: E402
from build_vll_dataset import photometric, downscale, target_stats  # noqa: E402

SAVE = 128
CROP_NATIVE = 256
NO_GESTURE = 13
SEVERITIES = (0.5, 0.8, 1.0)


def mine(shards_dir, out_dir, max_positives, min_score, seed=0,
         min_presence=0.5):
    """Mine detector crops from gesture-labelled 512px HaGRIDv2 shards.

    Every crop is immediately re-checked with the landmark model's presence head
    (a DIFFERENT network) and only confirmed hand crops are kept.  Without that
    check the label source is the detector's own output, and "the detector fires
    confidently but off the hand" is a measured failure mode: 41.6% of its top
    detections on gesture images were rejected by the arbiter, and training on
    them is what makes a verifier permissive (see VERIFIER_V2.md).
    """
    shards = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    if not shards:
        raise SystemExit(f"no parquet shards in {shards_dir}")
    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    lm = HandLandmarks()
    lm.load_weights(os.path.join(ML, "HandLandmarks.pth"))
    lm.eval()

    def presence(crop128):
        img = cv2.resize(crop128, (256, 256), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None]
                             ).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.no_grad():
            pr, _h, _r = lm(x)
        return float(pr.reshape(-1)[0])

    rows = []
    stats = dict(images=0, gesture_images=0, fired=0, skipped_no_gesture=0,
                 boxes=0, rejected_by_arbiter=0)
    for si, path in enumerate(shards):
        if len(rows) >= max_positives:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64):
            for rec in batch.to_pylist():
                if len(rows) >= max_positives:
                    break
                stats["images"] += 1
                label = int(rec["label"])
                if label == NO_GESTURE:
                    stats["skipped_no_gesture"] += 1
                    continue
                stats["gesture_images"] += 1
                buf = np.frombuffer(rec["image"]["bytes"], dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                fh, fw = img.shape[:2]
                padded, scale, left, top = letterbox(img, 256)
                with torch.no_grad():
                    dets = det.predict_on_image(padded)
                dets = [d for d in (dets[0] if dets else [])
                        if float(d[18]) >= min_score]
                if not dets:
                    continue
                stats["fired"] += 1
                stats["boxes"] += len(dets)
                d = max(dets, key=lambda x: float(x[18]))
                box = box_to_pixels(d[:4].tolist(), scale, left, top)
                kp0 = ((float(d[4]) * 256 - left) / scale,
                       (float(d[5]) * 256 - top) / scale)
                kp2 = ((float(d[8]) * 256 - left) / scale,
                       (float(d[9]) * 256 - top) / scale)
                cx, cy, side, rot = hand_roi(box, kp0, kp2)
                crop, _ = warp_rect(img, rotated_rect_to_points(cx, cy, side, side, rot),
                                    CROP_NATIVE)
                small = cv2.resize(crop, (SAVE, SAVE), interpolation=cv2.INTER_AREA)
                if min_presence > 0 and presence(small) < min_presence:
                    stats["rejected_by_arbiter"] += 1
                    continue
                name = f"h512_{si:04d}_{len(rows):06d}.png"
                dst = os.path.join(out_dir, "crops", "hand", name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, small)
                rows.append(dict(crop=os.path.join("crops", "hand", name),
                                 label="hand", source="hagrid512", held_out=0,
                                 iou="", score=round(float(d[18]), 4),
                                 box_w_frac=round((box[2] - box[0]) / fw, 4),
                                 gesture=str(label)))
        print(f"   {os.path.basename(path):<40} kept so far {len(rows)}", flush=True)

    checked = len(rows) + stats["rejected_by_arbiter"]
    print(f"\nmined {len(rows)} ARBITER-CONFIRMED crops from {len(shards)} shards:")
    print(f"   images seen        : {stats['images']}")
    print(f"   gesture images     : {stats['gesture_images']} "
          f"({stats['skipped_no_gesture']} no_gesture skipped)")
    print(f"   detector fired on  : {stats['fired']} "
          f"({100*stats['fired']/max(stats['gesture_images'],1):.1f}% of gesture "
          f"images)  <- sanity spot-check")
    if checked:
        print(f"   arbiter rejected   : {stats['rejected_by_arbiter']} "
              f"({100*stats['rejected_by_arbiter']/checked:.1f}% of the detector's "
              f"top detections were NOT confirmed as hand crops)")
    sizes = [r["box_w_frac"] for r in rows]
    if sizes:
        print(f"   box width (frac of image): median {np.median(sizes):.3f} "
              f"p10 {np.percentile(sizes,10):.3f} p90 {np.percentile(sizes,90):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(HERE, "hagrid512"))
    ap.add_argument("--base", default="verifier_dataset")
    ap.add_argument("--vll", default="vll_dataset")
    ap.add_argument("--out", default="big_dataset")
    ap.add_argument("--max-hagrid", type=int, default=60000)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--min-presence", type=float, default=0.5,
                    help="landmark-arbiter presence floor for a mined crop "
                         "to be kept as a hand (0 disables the check)")
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--neg-copies", type=int, default=4,
                    help="augmentation copies for CLUTTER (the scarce class)")
    ap.add_argument("--skip-mining", action="store_true")
    ap.add_argument("--mined-csv", default=None,
                    help="reuse an existing mined-crop manifest instead of "
                         "mining again (e.g. mined_filtered.csv, produced by "
                         "filter_mined.py after the landmark-arbiter check)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = os.path.join(HERE, args.out)
    # SAFETY: never wipe a directory that holds an input we are about to read.
    # (An earlier version rmtree'd --out before reading --mined-csv from inside
    # it and destroyed a 178k-crop dataset. Inputs are read FIRST, and only the
    # crops/ + manifest are cleared rather than the whole tree.)
    preloaded = None
    if args.mined_csv or args.skip_mining:
        src_csv = args.mined_csv or os.path.join(args.out, "mined.csv")
        src_path = src_csv if os.path.isabs(src_csv) else os.path.join(HERE, src_csv)
        preloaded = list(csv.DictReader(open(src_path)))
        print(f"reusing {len(preloaded)} mined crops from {src_csv}")
    os.makedirs(out, exist_ok=True)
    for sub in ("crops",):
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
    rows = []
    if preloaded is not None:
        rows += preloaded

    # ---- 1. the new HaGRIDv2 512px volume -------------------------------
    if preloaded is None:
        print("== mining the new HaGRIDv2 512px shards ==")
        mined = mine(args.shards, out, args.max_hagrid, args.min_score,
                     args.seed, args.min_presence)
        rows += mined
        with open(os.path.join(out, "mined.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mined[0].keys()))
            w.writeheader()
            w.writerows(mined)

    # ---- 2. existing base crops (train + held-out, unchanged) -----------
    base = list(csv.DictReader(open(os.path.join(HERE, args.base, "manifest.csv"))))
    # The HaGRID held-out splits are identified by SOURCE in the original build,
    # not by the held_out column, so map them explicitly -- otherwise they would
    # silently become training data.
    BASE_HELD = {"hagrid_valid", "hagrid_test", "own_nohand"}
    for r in base:
        src = os.path.join(HERE, args.base, r["crop"])
        if not os.path.isfile(src):
            continue
        name = "base_" + os.path.basename(r["crop"])
        sub = "clutter" if r["label"] == "clutter" else "hand"
        dst = os.path.join(out, "crops", sub, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        held = "1" if (r["source"] in BASE_HELD or r.get("held_out") == "1") else "0"
        rows.append(dict(crop=os.path.join("crops", sub, name), label=r["label"],
                         source=r["source"], held_out=held, iou="",
                         score=r.get("score", ""), box_w_frac=r.get("box_w_frac", ""),
                         gesture=r.get("gesture", "")))

    # ---- 3. VeryLowLight real crops (train + held-out, unchanged) -------
    vll = list(csv.DictReader(open(os.path.join(HERE, args.vll, "manifest.csv"))))
    for r in vll:
        if not r["source"].startswith("vll_real"):
            continue
        src = os.path.join(HERE, args.vll, r["crop"])
        if not os.path.isfile(src):
            continue
        sub = "clutter" if r["label"] == "clutter" else "hand"
        name = "vll_" + os.path.basename(r["crop"])
        dst = os.path.join(out, "crops", sub, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append(dict(crop=os.path.join("crops", sub, name), label=r["label"],
                         source=r["source"], held_out=r["held_out"], iou="",
                         score=r.get("score", ""), box_w_frac=r.get("box_w_frac", ""),
                         gesture=""))

    # ---- 4. augmentation, applied to BOTH classes, TRAIN sources only ---
    t_mean, t_std, n_frames = target_stats("VeryLowLight.mp4")
    print(f"\ntarget photometric statistics (VeryLowLight, {n_frames} frames): "
          f"mean BGR {t_mean.round(1)} std {t_std.round(1)}")
    aug = 0
    for r in list(rows):
        if r["held_out"] == "1" or r["label"] == "clutter" and r["source"] == "own_nohand":
            continue                       # never augment held-out crops
        if r["source"] in ("hagrid_valid", "hagrid_test"):
            continue
        img = cv2.imread(os.path.join(out, r["crop"]))
        if img is None:
            continue
        # Clutter is the scarce class: the new HaGRID volume adds tens of
        # thousands of positives, and a 1:1 balanced sampler would simply throw
        # most of them away.  Augmenting the negatives harder (same transforms,
        # same discipline, just more copies) is what keeps both classes usable.
        copies = args.neg_copies if r["label"] == "clutter" else args.aug_copies
        for sev in SEVERITIES:
            for k in range(copies):
                a = photometric(img, t_mean, t_std,
                                float(np.clip(sev + rng.uniform(-0.1, 0.1), 0, 1)),
                                rng)
                a = downscale(a, 0.2, 1.0, rng)
                sub = "clutter" if r["label"] == "clutter" else "hand"
                name = f"aug_{sev:.1f}_{k}_{os.path.basename(r['crop'])}"
                cv2.imwrite(os.path.join(out, "crops", sub, name), a)
                rows.append(dict(crop=os.path.join("crops", sub, name),
                                 label=r["label"],
                                 source=r["source"] + "_aug", held_out=0, iou="",
                                 score="", box_w_frac=r.get("box_w_frac", ""),
                                 gesture=""))
                aug += 1
    print(f"augmented copies written: {aug} (severities {SEVERITIES}, "
          f"downscale 0.2-1.0, both classes; clutter x{args.neg_copies}, "
          f"hand x{args.aug_copies})")

    with open(os.path.join(out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"\nwrote {len(rows)} crops -> {out}")
    c = Counter((r["label"], r["source"]) for r in rows)
    for (label, src), cnt in sorted(c.items()):
        ho = sum(1 for r in rows if r["label"] == label and r["source"] == src
                 and r["held_out"] == "1")
        print(f"   {label:<8} {src:<28} {cnt:>7}  (held out {ho})")


if __name__ == "__main__":
    main()
