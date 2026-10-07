"""Step 2.2/2.3: build a blob-pipeline-specific verifier and measure the gap.

Step 2.1 showed the mismatch is real: for the SAME hands, blob-pipeline crops score
1.95x higher than the training-style crops the current verifier was trained on,
and the 8-way max search inflates off-hand (moving clutter) crops more than
on-hand ones, which is what flips the separation negative.

So this trains a verifier on crops built through the ACTUAL blob pipeline (ROI =
blob side x MOTION_BLOB_ROI_SCALE, all 8 orientations as separate samples sharing
the blob's label), labelled by the MediaPipe arbiter, and evaluates it on
HELD-OUT frames with three inference rules (max of 8, mean of 8, single
orientation) so the selection-process question is answered with numbers rather
than assumed.

Negative sources: off-hand blobs in clips that contain a hand, PLUS blobs from a
synthetic moving-clutter clip (NohandLight with an injected translating patch) --
the exact failure mode the previous blob path had.

Usage:
    .venv-blazepalm\\Scripts\\python.exe blob_verifier.py
"""
import argparse
import csv
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (MotionGate, rotated_rect_to_points,  # noqa: E402
                               warp_rect)
from crop_verifier import SmallCNN, SIZE, make_verifier     # noqa: E402

DATASET = "blob_dataset"
CLIPS_GT = [("CloseLightMoving.mp4", "gt_clips.npz", None),
            ("CloseLightStill.mp4", "gt_clips.npz", None),
            ("CloseDarkStill.mp4", "gt_clips.npz", None),
            ("FarLight.mp4", "gt_clips.npz", None),
            ("FarDark.mp4", "gt_clips.npz", None),
            ("NohandLight.mp4", "gt_clips.npz", None),
            ("VeryLowLight.mp4", "gt_verylowlight.npz", "static_conf0.3")]
BLOCK, HOLDOUT_EVERY = 50, 3
SAVE = 128
CROP_NATIVE = 256


def is_holdout(n):
    return (n // BLOCK) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1


def hands_near(gt, n, window=5):
    """All MediaPipe hand positions near frame n (borrowed across gaps)."""
    out = []
    row = gt[gt[:, 0] == n]
    if len(row) and row[0, 1]:
        out.append((float(row[0, 2]), float(row[0, 3])))
    else:
        present = gt[gt[:, 1] == 1][:, 0]
        if len(present):
            near = present[np.abs(present - n) <= window]
            if len(near):
                k = near[np.argmin(np.abs(near - n))]
                row = gt[gt[:, 0] == k]
                out.append((float(row[0, 2]), float(row[0, 3])))
    return out


def collect(clip, gt, frames=100000, every=2, inject=None):
    """Blobs with labels, crops saved at every orientation."""
    out_dir = os.path.join(HERE, DATASET)
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    gate = MotionGate()
    rows = []
    patch = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or n >= frames:
            break
        n += 1
        fh, fw = frame.shape[:2]
        if inject is not None:                # synthetic moving-clutter clip
            if patch is None:
                w, h = int(0.15 * fw), int(0.15 * fh)
                py, px = int(0.25 * fh), int(0.10 * fw)
                patch = frame[py:py + h, px:px + w].copy()
            x = int((inject * n) % max(fw - w, 1))
            y = int(0.25 * fh)
            frame = frame.copy()
            frame[y:y + h, x:x + w] = patch
        gate.push(frame)
        if n % every:
            continue
        blobs = gate.blobs(frame.shape)
        if not blobs:
            continue
        hs = hands_near(gt, n) if gt is not None else []
        for cx, cy, side, _af in blobs:
            if inject is not None:
                label, src = 0, "injected_moving_clutter"
            else:
                near = min((float(np.hypot(cx - h[0], cy - h[1])) / fw
                            for h in hs), default=None)
                if near is None:
                    continue                  # no MediaPipe opinion: skip
                if near <= 0.12:
                    label, src = 1, clip.replace(".mp4", "")
                elif near > 0.20:
                    label, src = 0, clip.replace(".mp4", "")
                else:
                    continue                  # ambiguous band
            r = max(side * hmc.MOTION_BLOB_ROI_SCALE, 8.0)
            for k in range(hmc.MOTION_BLOB_ROTATIONS):
                rot = math.pi * k / hmc.MOTION_BLOB_ROTATIONS
                crop, _ = warp_rect(frame, rotated_rect_to_points(cx, cy, r, r, rot),
                                    CROP_NATIVE)
                sub = "hand" if label else "clutter"
                name = f"{src}_{n:05d}_{int(cx):04d}_{k}.png"
                d = os.path.join(out_dir, "crops", sub)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, name),
                            cv2.resize(crop, (SAVE, SAVE),
                                       interpolation=cv2.INTER_AREA))
                rows.append(dict(crop=os.path.join("crops", sub, name),
                                 label="hand" if label else "clutter",
                                 source=src, held_out=1 if is_holdout(n) else 0,
                                 frame=n, rot=k))
    cap.release()
    return rows


def build(args):
    out_dir = os.path.join(HERE, DATASET)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for clip, gtf, key in CLIPS_GT:
        z = np.load(os.path.join(HERE, gtf))
        gt = z[key] if key else z[clip]
        rows = collect(clip, gt, frames=args.frames_per_clip, every=args.every)
        all_rows += rows
        h = sum(1 for r in rows if r["label"] == "hand")
        print(f"   {clip:<22} {len(rows):>6} crops  ({h} hand / "
              f"{len(rows)-h} clutter)")
    if args.inject:
        z = np.load(os.path.join(HERE, "gt_clips.npz"))
        rows = collect("NohandLight.mp4", None, frames=args.frames_per_clip,
                       every=args.every, inject=8.0)
        all_rows += rows
        print(f"   {'injected moving clutter':<22} {len(rows):>6} crops  "
              f"(0 hand / {len(rows)} clutter)")
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} blob crops -> {out_dir}")
    return all_rows


# ---------------------------------------------------------------- training
def load(dataset):
    rows = list(csv.DictReader(open(os.path.join(dataset, "manifest.csv"))))
    X, y, meta = [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(dataset, r["crop"]))
        if img is None:
            continue
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y.append(1 if r["label"] == "hand" else 0)
        meta.append(r)
    return np.stack(X).astype(np.uint8), np.array(y, dtype=np.float32), meta


def score_all(model, X, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float() / 255.0
            out.append(torch.sigmoid(model(b)).numpy())
    return np.concatenate(out)


def group_by_blob(meta, p):
    """Collapse the 8 orientation samples of each blob into max/mean/single."""
    groups = {}
    for i, m in enumerate(meta):
        key = (m["source"], m["frame"], m["crop"].rsplit("_", 1)[0])
        groups.setdefault(key, []).append((int(m["rot"]), float(p[i]),
                                           m["label"], m["held_out"]))
    rows = []
    for key, items in groups.items():
        scores = {r: s for r, s, _l, _h in items}
        rows.append(dict(source=items[0][2] and key[0], label=items[0][2],
                         held_out=items[0][3],
                         mx=max(s for _r, s, _l, _h in items),
                         mean=float(np.mean([s for _r, s, _l, _h in items])),
                         single=scores.get(0, float("nan"))))
    return rows


def evaluate(tag, model, X, y, meta, threshold):
    p = score_all(model, X)
    rows = group_by_blob(meta, p)
    ho = [r for r in rows if r["held_out"] == "1"]
    print(f"\n== {tag}: held-out blob groups (n={len(ho)}) ==")
    print(f"   {'rule':<10}{'on-hand acc':>13}{'off-hand acc':>14}{'separation':>12}")
    for rule in ("mx", "mean", "single"):
        on = np.array([r[rule] for r in ho if r["label"] == "hand"])
        off = np.array([r[rule] for r in ho if r["label"] == "clutter"])
        on = on[~np.isnan(on)]
        off = off[~np.isnan(off)]
        if not len(on) or not len(off):
            continue
        a_on = 100 * float((on >= threshold).mean())
        a_off = 100 * float((off >= threshold).mean())
        print(f"   {rule:<10}{a_on:>12.1f}%{a_off:>13.1f}%{a_on - a_off:>11.1f}")
        if rule == "mx":
            print(f"      medians: on {np.median(on):.3f}  off "
                  f"{np.median(off):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-clip", type=int, default=400)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--out", default="blob_verifier.pt")
    ap.add_argument("--inject", action="store_true", default=True)
    ap.add_argument("--no-inject", dest="inject", action="store_false")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("== building blob-pipeline crops ==")
    if args.rebuild or not os.path.isfile(os.path.join(HERE, DATASET, "manifest.csv")):
        build(args)
    X, y, meta = load(os.path.join(HERE, DATASET))
    ho = np.array([m["held_out"] == "1" for m in meta])
    print(f"loaded {len(X)} blob crops: {int(y.sum())} hand / "
          f"{int((1-y).sum())} clutter ({int(ho.sum())} held out)")

    rng = np.random.default_rng(0)
    ip, ineg = np.where(~ho & (y == 1))[0], np.where(~ho & (y == 0))[0]
    n = min(len(ip), len(ineg))
    if n == 0:
        raise SystemExit("no training samples -- rerun with --rebuild")
    keep = np.concatenate([rng.choice(ip, n, replace=False),
                           rng.choice(ineg, n, replace=False)])
    rng.shuffle(keep)
    print(f"training on {len(keep)} crops ({n} hand / {n} clutter), "
          f"{args.epochs} epochs")

    torch.manual_seed(0)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X[keep]).permute(0, 3, 1, 2).float() / 255.0
    yt = torch.from_numpy(y[keep])
    bs = 128
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(keep))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"  epoch {ep+1}/{args.epochs} loss={tot/len(keep):.4f}", flush=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}")

    evaluate("BLOB-SPECIFIC verifier", model, X, y, meta, args.threshold)
    old = SmallCNN()
    old.load_state_dict(torch.load(os.path.join(HERE, "verifier_cnn.pt"),
                                   map_location="cpu", weights_only=True))
    old.eval()
    evaluate("existing verifier_cnn.pt on the same crops", old, X, y, meta,
             args.threshold)


if __name__ == "__main__":
    main()
