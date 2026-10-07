"""Train and evaluate the hand-vs-clutter verifier.

Train  : HaGRID crops + our own hand crops (source=hagrid_train / own_clip)
Test A : HaGRID held-out splits (hagrid_valid / hagrid_test)
Test B : **our own room's clutter** (source=own_nohand, held out) -- the real
         target.  Also compares against what the CURRENT 0.04 in-window size
         gate would have let through, so the improvement is explicit.

Usage: python train_verifier.py --dataset verifier_dataset [--out verifier.pt]
"""
import argparse
import csv
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline_classifier import (crop_features, FEATURE_NAMES,   # noqa: E402
                                 BaselineClassifier)

SIZE_GATE = 0.04        # MIN_BOX_FRAC_TRACK from hand_mouse_cursor.py


def load(dataset):
    rows = list(csv.DictReader(open(os.path.join(dataset, "manifest.csv"))))
    X, y, meta = [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(dataset, r["crop"]))
        if img is None:
            continue
        X.append(crop_features(img))
        y.append(1 if r["label"] == "hand" else 0)
        meta.append(r)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta


def auc(scores, labels):
    labels = np.asarray(labels)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="verifier_dataset")
    ap.add_argument("--out", default="verifier_lr.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, meta = load(args.dataset)
    src = np.array([m["source"] for m in meta])
    bw = np.array([float(m["box_w_frac"] or 0.0) for m in meta])
    print(f"loaded {len(X)} crops: {int(y.sum())} hand / {int((1-y).sum())} clutter")

    tr = np.isin(src, ["hagrid_train", "own_clip"])
    te_h = np.isin(src, ["hagrid_valid", "hagrid_test"])
    own_neg = src == "own_nohand"
    own_pos = src == "own_clip"

    # balance the training classes by subsampling the majority
    rng = np.random.default_rng(args.seed)
    idx_tr_pos = np.where(tr & (y == 1))[0]
    idx_tr_neg = np.where(tr & (y == 0))[0]
    n = min(len(idx_tr_pos), len(idx_tr_neg))
    keep = np.concatenate([rng.choice(idx_tr_pos, n, replace=False),
                           rng.choice(idx_tr_neg, n, replace=False)])
    print(f"training on {len(keep)} balanced crops "
          f"({n} hand / {n} clutter); HaGRID held-out {int(te_h.sum())}, "
          f"own-room held-out {int(own_neg.sum())} clutter")

    clf = BaselineClassifier(X.shape[1]).fit(X[keep], y[keep], epochs=3000)
    clf.save(args.out)
    print(f"saved -> {args.out}\n")

    p = clf.predict_proba(X)
    print("== separation (ROC-AUC) ==")
    print(f"  HaGRID held-out (valid+test) : {auc(p[te_h], y[te_h]):.4f}")
    print(f"  own-room hand vs clutter     : "
          f"{auc(p[own_pos | own_neg], y[own_pos | own_neg]):.4f}")

    print(f"\n== at threshold 0.50 ==")
    print(f"  HaGRID held-out hand kept    : {100*(p[te_h & (y==1)]>0.5).mean():5.1f}%  "
          f"(n={int((te_h&(y==1)).sum())})")
    print(f"  HaGRID held-out clutter cut  : {100*(p[te_h & (y==0)]<0.5).mean():5.1f}%  "
          f"(n={int((te_h&(y==0)).sum())})")
    print(f"  own-room HAND kept           : {100*(p[own_pos]>0.5).mean():5.1f}%  "
          f"(n={int(own_pos.sum())}, in-sample)")
    own_rej = 100 * (p[own_neg] < 0.5).mean()
    print(f"  own-room CLUTTER cut         : {own_rej:5.1f}%  "
          f"(n={int(own_neg.sum())}, HELD OUT)")

    gate_pass = 100 * (bw[own_neg] >= SIZE_GATE).mean()
    print(f"\n== the current {SIZE_GATE} size gate on the same own-room clutter ==")
    print(f"  clutter the {SIZE_GATE} gate would ACCEPT : {gate_pass:5.1f}% "
          f"(i.e. it cuts only {100-gate_pass:.1f}%)")

    print("\n== per-feature means ==")
    for j, f in enumerate(FEATURE_NAMES):
        print(f"  {f:<14} hand={X[own_pos | te_h][y[own_pos | te_h]==1][:, j].mean():8.4f}  "
              f"clutter={X[te_h | own_neg][y[te_h | own_neg]==0][:, j].mean():8.4f}  "
              f"| own_room_clutter={X[own_neg][:, j].mean():8.4f}")


if __name__ == "__main__":
    main()
