"""Train a small CNN hand-vs-clutter verifier and compare it with the logistic
baseline on exactly the same splits (including our own room's held-out clutter).

Usage: python train_cnn_verifier.py --dataset verifier_dataset [--epochs 8]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crop_verifier import SmallCNN, SIZE          # noqa: E402

SIZE_GATE = 0.04


def load(dataset):
    rows = list(csv.DictReader(open(os.path.join(dataset, "manifest.csv"))))
    X, y, meta = [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(dataset, r["crop"]))
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y.append(1 if r["label"] == "hand" else 0)
        meta.append(r)
    return (np.stack(X).astype(np.uint8), np.array(y, dtype=np.float32), meta)


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


def predict(model, X, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float() / 255.0
            out.append(torch.sigmoid(model(b)).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="verifier_dataset")
    ap.add_argument("--out", default="verifier_cnn.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, meta = load(args.dataset)
    src = np.array([m["source"] for m in meta])
    bw = np.array([float(m["box_w_frac"] or 0.0) for m in meta])
    print(f"loaded {len(X)} crops at {SIZE}x{SIZE}: "
          f"{int(y.sum())} hand / {int((1-y).sum())} clutter")

    tr = np.isin(src, ["hagrid_train", "own_clip"])
    te_h = np.isin(src, ["hagrid_valid", "hagrid_test"])
    own_neg = src == "own_nohand"
    own_pos = src == "own_clip"

    rng = np.random.default_rng(args.seed)
    ip, ineg = np.where(tr & (y == 1))[0], np.where(tr & (y == 0))[0]
    n = min(len(ip), len(ineg))
    keep = np.concatenate([rng.choice(ip, n, replace=False),
                           rng.choice(ineg, n, replace=False)])
    rng.shuffle(keep)
    print(f"training on {len(keep)} balanced crops for {args.epochs} epochs")

    torch.manual_seed(args.seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X[keep]).permute(0, 3, 1, 2).float() / 255.0
    yt = torch.from_numpy(y[keep])
    bs = 64
    t0 = time.time()
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
        print(f"  epoch {ep+1}/{args.epochs} loss={tot/len(keep):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}\n")

    p = predict(model, X)
    print("== small CNN ==")
    print(f"  HaGRID held-out AUC          : {auc(p[te_h], y[te_h]):.4f}")
    print(f"  own-room hand vs clutter AUC : "
          f"{auc(p[own_pos | own_neg], y[own_pos | own_neg]):.4f}")
    print(f"  HaGRID hand kept / clutter cut: "
          f"{100*(p[te_h&(y==1)]>0.5).mean():.1f}% / {100*(p[te_h&(y==0)]<0.5).mean():.1f}%")
    print(f"  own-room HAND kept           : {100*(p[own_pos]>0.5).mean():.1f}% "
          f"(n={int(own_pos.sum())}, in-sample)")
    print(f"  own-room CLUTTER cut         : {100*(p[own_neg]<0.5).mean():.1f}% "
          f"(n={int(own_neg.sum())}, HELD OUT)")
    print(f"  [{SIZE_GATE} size gate on the same clutter: cuts "
          f"{100*(bw[own_neg]<SIZE_GATE).mean():.1f}%]")


if __name__ == "__main__":
    main()
