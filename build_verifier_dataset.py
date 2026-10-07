"""Build the hand-vs-clutter verifier dataset.

Positives/negatives are mined by running OUR palm detector on HaGRIDv2 images
and comparing each detection with the MediaPipe palm box that ships with each
image:

    IoU(detection, label) >= HIT_IOU      -> 'hand'     (a real hand)
    IoU(detection, label) <= MISS_IOU     -> 'clutter'  (the detector fired on
                                            the background -- exactly the
                                            poster/corner failure mode)
    in between                            -> skipped (ambiguous)

Crops are the same rotation-normalised 256->128 crops the live pipeline feeds
the classifier, so there is no train/serve skew.

Our own recordings are ingested too:
  * clips whose crops are real hands   -> 'hand'
  * Nohand* clips                      -> 'clutter', and are marked
                                          held_out=True so they are NEVER
                                          trained on (they are the real-room
                                          evaluation set).

Usage:
    python build_verifier_dataset.py --hagrid-dir hagrid --own dataset_standin \
        --out verifier_dataset --splits train valid test
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)

from blazepalm import PalmDetector                        # noqa: E402
from handlandmarks import HandLandmarks                   # noqa: E402
from hand_pipeline import (letterbox, box_to_pixels, hand_roi,   # noqa: E402
                           rotated_rect_to_points, warp_rect)

HIT_FRAC = 0.5      # detection centre within this x label size -> it IS the hand
MISS_FRAC = 1.5     # centre beyond this x label size -> detector fired on background
SAVE_SIZE = 128     # crop size written to disk (features use this)
CROP_NATIVE = 256   # detector's own input size

# NOTE on why this is centre-based, not IoU-based: our detector's decoded box is
# the 2.6x-expanded palm ROI, while HaGRID ships the raw palm box.  IoU between
# a centred pair is therefore only ~0.15, which would mislabel almost every
# true positive.  Centre distance is scale-invariant and avoids that entirely.


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def detect_crops(detector, frame):
    """All detections (no size gate) as (box, score, crop256, rotation)."""
    padded, scale, left, top = letterbox(frame, 256)
    with torch.no_grad():
        dets = detector.predict_on_image(padded)
    out = []
    if not dets:
        return out
    for d in dets[0]:
        box = box_to_pixels(d[:4].tolist(), scale, left, top)
        kps = [((float(d[4 + 2 * k]) * 256 - left) / scale,
                (float(d[5 + 2 * k]) * 256 - top) / scale) for k in range(7)]
        cx, cy, side, rot = hand_roi(box, kps[0], kps[2])
        rect = rotated_rect_to_points(cx, cy, side, side, rot)
        crop, _ = warp_rect(frame, rect, CROP_NATIVE)
        out.append((box, float(d[18]), crop, rot))
    return out


def add_sample(rows, out_dir, label, crop, source, meta, held_out=False):
    d = os.path.join(out_dir, "crops", label)
    os.makedirs(d, exist_ok=True)
    name = f"{source}__{len(rows):06d}.png"
    cv2.imwrite(os.path.join(d, name),
                cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA))
    r = dict(crop=os.path.join("crops", label, name), label=label, source=source,
             held_out=int(held_out))
    r.update(meta)
    rows.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hagrid-dir", default="hagrid")
    ap.add_argument("--own", default="dataset_standin")
    ap.add_argument("--out", default="verifier_dataset")
    ap.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    ap.add_argument("--max-per-split", type=int, default=4000)
    ap.add_argument("--max-neg-per-split", type=int, default=4000)
    args = ap.parse_args()

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    lm_model = HandLandmarks()
    lm_model.load_weights(os.path.join(ML, "HandLandmarks.pth"))
    lm_model.eval()

    rows = []
    os.makedirs(args.out, exist_ok=True)

    # ---------------- HaGRID ----------------
    for split in args.splits:
        files = [f for f in sorted(os.listdir(args.hagrid_dir))
                 if f.startswith(split) and f.endswith(".parquet")]
        if not files:
            print(f"  no parquet for split '{split}'")
            continue
        import pyarrow.parquet as pq
        n_pos = n_neg = n_skip = n_img = 0
        for fn in files:
            pf = pq.ParquetFile(os.path.join(args.hagrid_dir, fn))
            for batch in pf.iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    n_img += 1
                    buf = np.frombuffer(r["image"]["bytes"], np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    labels = []          # (box_px, palm7_px)
                    for inst in json.loads(r["instances"]):
                        bb = inst.get("palm_bbox_xyxy")
                        if not bb:
                            continue
                        box = (bb[0] * w, bb[1] * h, bb[2] * w, bb[3] * h)
                        kps = inst.get("palm7_keypoints") or []
                        pts = []
                        for kp in kps:
                            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                                pts.append((float(kp[0]) * w, float(kp[1]) * h))
                        labels.append((box, pts))
                    if not labels:
                        continue

                    # (A) label-derived positive: the crop the pipeline WOULD
                    # produce if it detected the hand there.  Guarantees a
                    # well-centred positive per image regardless of whether our
                    # detector actually fires on this (often small) hand.
                    lb, lpts = labels[0]
                    lcx, lcy = (lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0
                    lsize = max(lb[2] - lb[0], lb[3] - lb[1])
                    if len(lpts) >= 3 and n_pos < args.max_per_split:
                        try:
                            # HaGRID ships the RAW palm box, but our detector's
                            # decoded box is the 2.6x-expanded ROI and hand_roi()
                            # returns side == box size.  Feeding the raw label
                            # box would therefore produce a crop 2.6x TIGHTER
                            # than any real detector crop, and the classifier
                            # would learn the wrong scale (measured: it then
                            # rejected 95% of our own hand crops).  Expand the
                            # label box into our convention first.
                            ex = (lb[2] - lb[0]) * 1.3      # x2.6/2
                            ey = (lb[3] - lb[1]) * 1.3
                            lb_our = (lcx - ex, lcy - ey, lcx + ex, lcy + ey)
                            cx, cy, side, rot = hand_roi(lb_our, lpts[0], lpts[2])
                            rect = rotated_rect_to_points(cx, cy, side, side, rot)
                            crop, _ = warp_rect(img, rect, CROP_NATIVE)
                            add_sample(rows, args.out, "hand", crop,
                                       f"hagrid_{split}",
                                       dict(iou="label", score="",
                                            box_w_frac=round(lsize / w, 4),
                                            gesture=r.get("gesture_label")))
                            n_pos += 1
                        except Exception:
                            pass

                    for box, score, crop, _rot in detect_crops(det, img):
                        bcx, bcy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                        d = float(np.hypot(bcx - lcx, bcy - lcy))
                        if d <= HIT_FRAC * lsize:
                            if n_pos >= args.max_per_split:
                                continue
                            add_sample(rows, args.out, "hand", crop, f"hagrid_{split}",
                                       dict(iou=round(d / lsize, 3),
                                            score=round(score, 4),
                                            box_w_frac=round((box[2] - box[0]) / w, 4),
                                            gesture=r.get("gesture_label")))
                            n_pos += 1
                        elif d >= MISS_FRAC * lsize:
                            if n_neg >= args.max_neg_per_split:
                                continue
                            add_sample(rows, args.out, "clutter", crop, f"hagrid_{split}",
                                       dict(iou=round(d / lsize, 3),
                                            score=round(score, 4),
                                            box_w_frac=round((box[2] - box[0]) / w, 4),
                                            gesture=r.get("gesture_label")))
                            n_neg += 1
                        else:
                            n_skip += 1
        print(f"  hagrid_{split}: {n_img} images -> {n_pos} hand, {n_neg} clutter, "
              f"{n_skip} ambiguous skipped")

    # ---------------- our own recordings ----------------
    own_man = os.path.join(args.own, "manifest.csv")
    if os.path.exists(own_man):
        pos_src = {"unlabeled"}          # real-hand clips in the stand-in set
        for r in csv.DictReader(open(own_man)):
            img = cv2.imread(os.path.join(args.own, r["crop"]))
            if img is None:
                continue
            if r["label"] == "no-hand":
                # OUR room's real clutter -> held out, never trained on
                add_sample(rows, args.out, "clutter", img, "own_nohand",
                           dict(iou="", score=r.get("score", ""),
                                box_w_frac=r.get("box_w_frac", ""),
                                gesture=""), held_out=True)
            elif r["label"] in pos_src:
                add_sample(rows, args.out, "hand", img, "own_clip",
                           dict(iou="", score=r.get("score", ""),
                                box_w_frac=r.get("box_w_frac", ""), gesture=""))

    man = os.path.join(args.out, "manifest.csv")
    fields = ["crop", "label", "source", "held_out", "iou", "score",
              "box_w_frac", "gesture"]
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    from collections import Counter
    c = Counter((r["label"], r["source"]) for r in rows)
    print(f"\nwrote {len(rows)} crops -> {args.out}")
    for k in sorted(c):
        print(f"   {k[0]:<8} {k[1]:<14} {c[k]}")
    held = sum(1 for r in rows if r["held_out"])
    print(f"held out (never trained on): {held}")
    print(f"manifest -> {man}")


if __name__ == "__main__":
    main()
