"""Step 1: dark/distance-heavy augmentation on the data already on disk.

No new raw data. This takes the arbiter-confirmed crop pool from `big_dataset`
(which is the source that already excludes the ~42% of mined crops whose boxes were
not on the hand) and generates many more synthetic variants aimed squarely at the
target condition -- far AND dark at the same time:

  * four photometric severities of the measured match to VeryLowLight's statistics
  * three distance bands: none / mild (0.45-1.0) / strong (0.12-0.35)
  * combined variants: severity and band are drawn independently, so most samples
    are dark *and* downscaled -- the actual target regime
  * the SAME variant procedure is applied to both classes, so brightness and
    sharpness cannot become class cues (the discipline used in every round)

Held-out rows are copied across untouched and never augmented, keeping the splits
identical to previous rounds so numbers stay comparable.

Usage:
    .venv-blazepalm\\Scripts\\python.exe build_dark_dataset.py
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_vll_dataset import photometric, downscale, target_stats  # noqa: E402

SEVERITIES = (0.35, 0.55, 0.75, 1.0)
BANDS = (("none", 1.0, 1.0, 0.15),
         ("mild", 0.45, 1.0, 0.35),
         ("strong", 0.12, 0.35, 0.50))


def make_variant(img, t_mean, t_std, rng):
    sev = float(np.clip(rng.choice(SEVERITIES) + rng.uniform(-0.08, 0.08), 0.0, 1.0))
    a = photometric(img, t_mean, t_std, sev, rng)
    names = [b[0] for b in BANDS]
    probs = np.array([b[3] for b in BANDS], dtype=float)
    probs /= probs.sum()
    band = str(rng.choice(names, p=probs))
    lo, hi = next((b[1], b[2]) for b in BANDS if b[0] == band)
    a = downscale(a, lo, hi, rng)
    return a, sev, band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="big_dataset")
    ap.add_argument("--out", default="dark_dataset")
    ap.add_argument("--pos-variants", type=int, default=2)
    ap.add_argument("--neg-variants", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_mean, t_std, n_frames = target_stats("VeryLowLight.mp4")
    print(f"target photometric statistics (VeryLowLight, {n_frames} frames): "
          f"mean BGR {t_mean.round(1)}  std {t_std.round(1)}")
    print(f"variants per crop: hand x{args.pos_variants}, clutter x{args.neg_variants}; "
          f"severities {SEVERITIES}; distance bands "
          f"{[(b[0], b[1], b[2]) for b in BANDS]}")

    base = list(csv.DictReader(open(os.path.join(HERE, args.base, "manifest.csv"))))
    out = os.path.join(HERE, args.out)
    if os.path.isdir(out):
        # SAFETY: only clear crops/ -- never the directory holding an input
        shutil.rmtree(os.path.join(out, "crops"), ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    rows = []
    n_aug = 0
    for r in base:
        src = os.path.join(HERE, args.base, r["crop"])
        if not os.path.isfile(src):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        sub = "clutter" if r["label"] == "clutter" else "hand"
        name = "orig_" + os.path.basename(r["crop"])
        dst = os.path.join(out, "crops", sub, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append(dict(crop=os.path.join("crops", sub, name), label=r["label"],
                         source=r["source"], held_out=r["held_out"], iou="",
                         score=r.get("score", ""), box_w_frac=r.get("box_w_frac", ""),
                         gesture=r.get("gesture", "")))
        if r["held_out"] == "1":
            continue                      # never augment held-out rows
        n = args.neg_variants if r["label"] == "clutter" else args.pos_variants
        for k in range(n):
            a, sev, band = make_variant(img, t_mean, t_std, rng)
            nm = f"dk_{k}_{os.path.basename(r['crop'])}"
            cv2.imwrite(os.path.join(out, "crops", sub, nm), a)
            rows.append(dict(crop=os.path.join("crops", sub, nm),
                             label=r["label"], source=r["source"] + "_dark",
                             held_out="0", iou="", score="",
                             box_w_frac=r.get("box_w_frac", ""), gesture=""))
            n_aug += 1

    with open(os.path.join(out, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter((r["label"], r["source"]) for r in rows)
    print(f"\nwrote {len(rows)} crops ({n_aug} new variants) -> {out}")
    tot = Counter()
    for (label, src), cnt in sorted(c.items()):
        ho = sum(1 for r in rows if r["label"] == label and r["source"] == src
                 and r["held_out"] == "1")
        print(f"   {label:<8} {src:<30} {cnt:>7}  (held out {ho})")
        tot[label] += cnt - ho
    print(f"\ntraining pool: {tot['hand']} hand / {tot['clutter']} clutter "
          f"-> balanced sampler will use {2*min(tot['hand'], tot['clutter'])} crops")


if __name__ == "__main__":
    main()
