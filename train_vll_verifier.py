"""Step 3: retrain the crop verifier on the VeryLowLight-augmented set.

Evaluates the new model and the previous one (verifier_cnn.pt) on exactly the
same held-out splits, so the comparison is like for like:

    VLL held-out hands        - real crops from VeryLowLight, blocks never trained on
    VLL held-out clutter      - real clutter from VeryLowLight, same held-out blocks
    HaGRID held-out (v+t)     - did we destroy daylight generalisation?
    own-room clutter (23)     - the original held-out target

Usage:
    .venv-blazepalm\\Scripts\\python.exe train_vll_verifier.py
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crop_verifier import SmallCNN, SIZE          # noqa: E402

THRESHOLD = 0.30          # hmc.VERIFIER_THRESHOLD


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
    return np.stack(X).astype(np.uint8), np.array(y, dtype=np.float32), meta


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


def report(tag, model, X, y, meta):
    p = predict(model, X)
    src = np.array([m["source"] for m in meta])
    ho = np.array([m["held_out"] == "1" for m in meta])
    vll_pos = ho & (src == "vll_real_mp")
    vll_pos_d = ho & (src == "vll_real_det")
    vll_neg = ho & (src == "vll_real_clutter")
    hag = ho & np.isin(src, ["hagrid_valid", "hagrid_test"])
    own = src == "own_nohand"

    print(f"\n== {tag} ==")
    for name, sel in (("VLL held-out hands (MediaPipe-anchored)", vll_pos),
                      ("VLL held-out hands (our detector recipe)", vll_pos_d),
                      ("VLL held-out clutter", vll_neg)):
        if sel.sum() == 0:
            continue
        pp = p[sel]
        if y[sel].sum() > 0 and (1 - y[sel]).sum() > 0:
            a = f"  AUC {auc(pp, y[sel]):.4f}"
        else:
            a = ""
        if y[sel].sum() > 0:
            kept = 100 * (pp[y[sel] == 1] >= THRESHOLD).mean()
        else:
            kept = float("nan")
        if (1 - y[sel]).sum() > 0:
            cut = 100 * (pp[y[sel] == 0] < THRESHOLD).mean()
        else:
            cut = float("nan")
        print(f"   {name:<44} n={int(sel.sum()):>5}  "
              f"kept {kept:5.1f}%  cut {cut:5.1f}%{a}")
    if hag.sum():
        print(f"   {'HaGRID held-out (valid+test)':<44} n={int(hag.sum()):>5}  "
              f"AUC {auc(p[hag], y[hag]):.4f}  "
              f"kept {100*(p[hag & (y==1)]>=THRESHOLD).mean():5.1f}%  "
              f"cut {100*(p[hag & (y==0)]<THRESHOLD).mean():5.1f}%")
    if own.sum():
        print(f"   {'own-room clutter (held out, n=23)':<44} n={int(own.sum()):>5}  "
              f"cut {100*(p[own]<THRESHOLD).mean():5.1f}%")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vll_dataset")
    ap.add_argument("--out", default="verifier_cnn_vll.pt")
    ap.add_argument("--old", default="verifier_cnn.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, meta = load(args.dataset)
    ho = np.array([m["held_out"] == "1" for m in meta])
    print(f"loaded {len(X)} crops at {SIZE}x{SIZE}: "
          f"{int(y.sum())} hand / {int((1-y).sum())} clutter "
          f"({int(ho.sum())} held out)")

    rng = np.random.default_rng(args.seed)
    ip, ineg = np.where(~ho & (y == 1))[0], np.where(~ho & (y == 0))[0]
    n = min(len(ip), len(ineg))
    keep = np.concatenate([rng.choice(ip, n, replace=False),
                           rng.choice(ineg, n, replace=False)])
    rng.shuffle(keep)
    print(f"training on {len(keep)} balanced crops ({n} hand / {n} clutter) "
          f"for {args.epochs} epochs")

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
    print(f"saved -> {args.out}")

    old = None
    if os.path.isfile(os.path.join(HERE, args.old)):
        old = SmallCNN()
        old.load_state_dict(torch.load(os.path.join(HERE, args.old),
                                       map_location="cpu", weights_only=True))
        old.eval()
        report(f"PREVIOUS model ({args.old})", old, X, y, meta)
    report(f"RETRAINED model ({args.out})", model, X, y, meta)


if __name__ == "__main__":
    main()
