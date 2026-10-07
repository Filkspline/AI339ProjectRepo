"""Compare verifier models at MATCHED precision, not at an arbitrary threshold.

The 130k-crop model is much more permissive at 0.30 than the 11k model, so the
head-to-head at a fixed threshold conflates "worse model" with "different
operating point".  This sweeps the threshold for each model on the standard
held-out splits and reports, for every threshold, what each model keeps/cuts --
so the two can be compared at equal clutter rejection.

Splits (identical to every previous training round):
    VLL held-out hands (MediaPipe-anchored / detector recipe)
    VLL held-out clutter
    HaGRID held-out (hagrid_valid + hagrid_test)
    own-room clutter (own_nohand, n=23)

Usage:
    .venv-blazepalm\\Scripts\\python.exe sweep_verifier_threshold.py
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crop_verifier import SmallCNN                     # noqa: E402
from train_vll_verifier import load, predict           # noqa: E402

THRESHOLDS = (0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="big_dataset")
    ap.add_argument("--models", nargs="+",
                    default=["verifier_cnn.pt", "verifier_cnn_big.pt"])
    args = ap.parse_args()

    X, y, meta = load(os.path.join(HERE, args.dataset))
    src = np.array([m["source"] for m in meta])
    ho = np.array([m["held_out"] == "1" for m in meta])
    vll_mp = ho & (src == "vll_real_mp")
    vll_det = ho & (src == "vll_real_det")
    vll_neg = ho & (src == "vll_real_clutter")
    hag = ho & np.isin(src, ["hagrid_valid", "hagrid_test"])
    own = src == "own_nohand"
    print(f"held-out crop counts: VLL hands {int(vll_mp.sum())} (MP-anchored) + "
          f"{int(vll_det.sum())} (detector recipe), VLL clutter "
          f"{int(vll_neg.sum())}, HaGRID {int(hag.sum())}, own-room clutter "
          f"{int(own.sum())}\n")

    for name in args.models:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            print(f"missing {name}")
            continue
        m = SmallCNN()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        p = predict(m, X)
        print(f"== {name} ==")
        print(f"   {'thr':>5}{'HaGRID AUC':>12}{'HaGRID cut':>12}"
              f"{'own cut':>9}{'VLLh kept(MP)':>15}{'VLLh kept(det)':>16}"
              f"{'VLLc cut':>10}")
        o = np.argsort(p[hag])
        ranks = np.empty(int(hag.sum()))
        ranks[o] = np.arange(1, int(hag.sum()) + 1)
        yh = y[hag]
        npos, nneg = yh.sum(), (1 - yh).sum()
        auc = ((ranks[yh == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
               if npos and nneg else float("nan"))
        for t in THRESHOLDS:
            print(f"   {t:>5.2f}{auc:>12.4f}"
                  f"{100*(p[hag & (y==0)] < t).mean():>11.1f}%"
                  f"{100*(p[own] < t).mean():>8.1f}%"
                  f"{100*(p[vll_mp] >= t).mean():>14.1f}%"
                  f"{100*(p[vll_det] >= t).mean():>15.1f}%"
                  f"{100*(p[vll_neg] < t).mean():>9.1f}%")
        print()

    # the headline comparison: at matched own-room clutter rejection
    print("== matched-precision comparison (interpolated to the 11k model's "
          "operating point) ==")
    models = {}
    for name in args.models:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            continue
        m = SmallCNN()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        models[name] = predict(m, X)
    if len(models) >= 2:
        ref_name = args.models[0]
        ref = models[ref_name]
        target = float((ref[own] < 0.30).mean())
        print(f"   reference: {ref_name} at 0.30 cuts {100*target:.1f}% of own-room "
              f"clutter")
        for name, p in models.items():
            # smallest threshold whose own-room cut >= the reference's
            best = None
            for t in np.arange(0.05, 1.0, 0.01):
                if (p[own] < t).mean() >= target:
                    best = t
                    break
            if best is None:
                print(f"   {name:<26} never reaches that clutter rejection")
                continue
            print(f"   {name:<26} needs thr {best:.2f} -> own cut "
                  f"{100*(p[own]<best).mean():.1f}%, HaGRID cut "
                  f"{100*(p[hag & (y==0)]<best).mean():.1f}%, VLL hands kept "
                  f"{100*(p[vll_mp]>=best).mean():.1f}% (MP-anchored) / "
                  f"{100*(p[vll_det]>=best).mean():.1f}% (detector), "
                  f"VLL clutter cut {100*(p[vll_neg]<best).mean():.1f}%")


if __name__ == "__main__":
    main()
