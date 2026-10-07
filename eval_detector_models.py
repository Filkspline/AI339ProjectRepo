"""Detector-only localisation comparison: original vs fine-tuned weights.

The metric is the one from VLL_DIAGNOSIS.md -- "detector-box-contains-hand": on
frames where lag-free MediaPipe sees a hand, does the detector's TOP detection
(score >= 0.5) have its box around MediaPipe's palm centre (15% slack)?  That is the
metric the 64.3% bottleneck was measured with, so it is the one to judge a
fine-tune by.

Cases:
  * VeryLowLight holdout blocks only (never trained on)   -> cross-frame generalisation
  * VeryLowLight holdout with the dark augmentation        -> the training regime
  * VeryLowLight train blocks                              -> the in-sample /
                                                              overfitting gap
  * the seven original clips, untouched by training        -> the light-condition trade

Usage:
    .venv-blazepalm\\Scripts\\python.exe eval_detector_models.py \
        --models palmdetector.pth palmdetector_dark.pth
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
from finetune_detector_dark import eval_clip, LIGHT_CLIPS   # noqa: E402
from build_vll_dataset import target_stats                  # noqa: E402


def load(path):
    d = PalmDetector()
    d.load_weights(path)
    d.load_anchors(os.path.join(ML, "anchors.npy"))
    d.eval()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["palmdetector.pth", "palmdetector_dark.pth"])
    ap.add_argument("--clip", default="VeryLowLight.mp4")
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--gt-key", default="static_conf0.3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_mean, t_std, _ = target_stats(args.clip)
    gt = np.load(os.path.join(HERE, args.gt))[args.gt_key]
    gtc = np.load(os.path.join(HERE, "gt_clips.npz"))

    dets = {}
    for m in args.models:
        if os.path.isabs(m):
            p = m
        else:
            # a bare filename may live in the project root OR next to the ML code
            p = os.path.join(HERE, m)
            if not os.path.isfile(p):
                p = os.path.join(ML, m)
        if not os.path.isfile(p):
            print(f"missing {m}")
            continue
        dets[os.path.basename(m)] = load(p)
    names = list(dets)
    if not names:
        raise SystemExit("no models loaded")

    cases = [
        ("VeryLowLight holdout (as recorded)", args.clip, gt,
         dict(holdout_only=True), "dark"),
        ("VeryLowLight holdout + dark aug", args.clip, gt,
         dict(holdout_only=True, dark_aug=True, rng=rng, t_mean=t_mean,
              t_std=t_std), "dark"),
        ("VeryLowLight train blocks (in-sample)", args.clip, gt,
         dict(holdout_only=False, dark_aug=True, rng=rng, t_mean=t_mean,
              t_std=t_std), "dark"),
    ]
    for clip in LIGHT_CLIPS:
        cases.append((clip, clip, gtc[f"static:{clip}"], dict(), "light"))

    print("detector-only localisation: box-contains-hand, top detection "
          "(score >= 0.5)\n")
    w = 22
    hdr = f"{'case':<40}{'hand frames':>12}" + "".join(f"{n:>{w}}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    res = {n: {} for n in names}
    kind_of = {}
    for label, clip, gt_, kw, kind in cases:
        line = f"{label:<40}"
        handed = 0
        for n in names:
            r = eval_clip(dets[n], clip, gt_, **kw)
            handed = r["handed"]
            res[n][label] = r["contain"]
            kind_of[label] = kind
            line += f"{(f'{100*r[chr(99)+chr(111)+chr(110)+chr(116)+chr(97)+chr(105)+chr(110)]:.1f}%'):>{w}}"
        print(f"{line[:40]}{handed:>12}" + line[40:])

    if len(names) >= 2:
        a, b = names[0], names[-1]
        dark = [l for l in res[a]
                if l.startswith("VeryLowLight holdout + dark")]
        light = [l for l in res[a] if kind_of[l] == "light"]
        tr = [l for l in res[a] if "train blocks" in l]
        print(f"\n   {b} vs {a}:")
        for group, labels in (("dark condition (holdout, dark-augmented)", dark),
                              ("dark condition (holdout, as recorded)",
                               [l for l in res[a] if "as recorded" in l]),
                              ("light conditions (7 clips, unseen)",
                               light)):
            if labels:
                d = np.mean([res[b][l] - res[a][l] for l in labels])
                print(f"     {group:<46} {100*d:+6.1f} points "
                      f"(over {len(labels)} clip(s))")
        if tr:
            gap_a = res[a][tr[0]] - res[a][dark[0]]
            gap_b = res[b][tr[0]] - res[b][dark[0]]
            print(f"\n   overfitting gap (train blocks - holdout, dark-augmented): "
                  f"{100*gap_a:+.1f} points original, {100*gap_b:+.1f} points "
                  f"fine-tuned")
    print("\n   The metric is measured only on frames where lag-free MediaPipe sees "
          "a hand,\n   so it is unaffected by how many false positives the detector "
          "emits.")


if __name__ == "__main__":
    main()
