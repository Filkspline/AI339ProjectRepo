"""Cheap baseline for the palm-crop classifier: interpretable crop statistics
-> logistic regression (implemented in torch, so no new dependency).

Two jobs, one model:
  * hand vs clutter  (the eventual replacement for the box-size FP gate)
  * open vs fist

Training is one call away once labeled footage exists:
    python baseline_classifier.py --dataset dataset --out baseline.pt
or from code:
    from baseline_classifier import train_from_directory
    clf = train_from_directory("dataset", "baseline.pt")

Smoke test against the stand-in dataset (mechanics + feature sanity only --
the stand-in clips have NO open/fist labels):
    python baseline_classifier.py --smoke
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

FEATURE_NAMES = ["fg_frac", "radial_mean", "radial_max", "radial_std",
                 "compactness", "edge_density", "mean_int", "std_int"]


# ---------------------------------------------------------------- features
def crop_features(crop_bgr):
    """Cheap, interpretable features from a rotation-normalised hand crop.

    Rationale: an OPEN hand has spread fingers -> lower foreground fill relative
    to its extent, larger radial reach, more internal edge structure.  A FIST is
    compact -> higher fill, smaller radial reach.  The same features also
    separate a real hand crop from a clutter crop.
    """
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = g.shape
    _, m = cv2.threshold((g * 255).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = m > 0
    fg_frac = float(fg.mean())

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2) / (0.5 * min(h, w))
    rr = r[fg]
    if rr.size:
        radial_mean, radial_max, radial_std = float(rr.mean()), float(rr.max()), float(rr.std())
    else:
        radial_mean = radial_max = radial_std = 0.0

    ys, xs = np.nonzero(fg)
    if ys.size:
        bbox = (ys.max() - ys.min() + 1.0) * (xs.max() - xs.min() + 1.0)
        compactness = float((fg.sum()) / bbox)
    else:
        compactness = 0.0

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(np.sqrt(gx * gx + gy * gy).mean())

    return np.array([fg_frac, radial_mean, radial_max, radial_std, compactness,
                     edge_density, float(g.mean()), float(g.std())], dtype=np.float32)


# ---------------------------------------------------------------- model
class BaselineClassifier:
    """Standardised logistic regression over crop features."""

    def __init__(self, n_features=len(FEATURE_NAMES)):
        self.n_features = n_features
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.std = np.ones(n_features, dtype=np.float32)
        self.linear = torch.nn.Linear(n_features, 1)

    def _norm(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y, epochs=1500, lr=0.05, weight_decay=1e-3, verbose=False):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Xt = torch.tensor(self._norm(X))
        yt = torch.tensor(y)
        opt = torch.optim.Adam(self.linear.parameters(), lr=lr,
                               weight_decay=weight_decay)
        lossf = torch.nn.BCEWithLogitsLoss()
        for i in range(epochs):
            opt.zero_grad()
            loss = lossf(self.linear(Xt).squeeze(-1), yt)
            loss.backward()
            opt.step()
            if verbose and (i + 1) % 500 == 0:
                print(f"    epoch {i+1}: loss={loss.item():.4f}")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.linear(torch.tensor(self._norm(X))).squeeze(-1)
        return torch.sigmoid(logits).numpy()

    def save(self, path):
        torch.save(dict(state=self.linear.state_dict(), mean=self.mean,
                        std=self.std, features=FEATURE_NAMES), path)

    @staticmethod
    def load(path):
        d = torch.load(path, weights_only=False)
        clf = BaselineClassifier(len(d["features"]))
        clf.linear.load_state_dict(d["state"])
        clf.mean, clf.std = d["mean"], d["std"]
        return clf


# ---------------------------------------------------------------- dataset IO
def load_dataset(dataset_dir, positive_labels=None, negative_labels=None,
                 max_per_label=None):
    """Read manifest.csv, load crops, return (X, y, meta).

    If positive/negative label sets are given, a binary problem is built from
    them (e.g. positives={'open','fist'}, negatives={'no-hand'}).
    Otherwise y is the 0/1 index of the label (multi-class not supported here).
    """
    man = os.path.join(dataset_dir, "manifest.csv")
    rows = list(csv.DictReader(open(man)))
    X, y, meta = [], [], []
    counts = {}
    for r in rows:
        lab = r["label"]
        if positive_labels is not None:
            if lab in positive_labels:
                target = 1
            elif lab in negative_labels:
                target = 0
            else:
                continue
        else:
            target = 0
        counts[lab] = counts.get(lab, 0) + 1
        if max_per_label and counts[lab] > max_per_label:
            continue
        img = cv2.imread(os.path.join(dataset_dir, r["crop"]))
        if img is None:
            continue
        X.append(crop_features(img))
        y.append(target)
        meta.append(r)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta


def train_from_directory(dataset_dir, out_path=None, positive_labels=None,
                         negative_labels=None, epochs=1500, verbose=True):
    """Single-call training entry point."""
    X, y, meta = load_dataset(dataset_dir, positive_labels, negative_labels)
    if len(X) == 0:
        raise SystemExit("no crops found -- is the manifest present and labeled?")
    clf = BaselineClassifier(X.shape[1]).fit(X, y, epochs=epochs, verbose=verbose)
    acc = float(((clf.predict_proba(X) > 0.5).astype(np.float32) == y).mean())
    if verbose:
        print(f"  trained on {len(X)} crops ({int(y.sum())} pos / {int((1-y).sum())} neg), "
              f"train acc={acc:.3f}")
    if out_path:
        clf.save(out_path)
        if verbose:
            print(f"  saved -> {out_path}")
    return clf


# ---------------------------------------------------------------- smoke test
def smoke(dataset_dir):
    print("== smoke test: feature sanity on stand-in crops ==")
    X, _, meta = load_dataset(dataset_dir)
    labels = np.array([m["label"] for m in meta])
    print(f"crops={len(X)}")
    for lab in sorted(set(labels)):
        sel = labels == lab
        print(f"\n  label '{lab}' (n={int(sel.sum())}):")
        for j, fname in enumerate(FEATURE_NAMES):
            v = X[sel, j]
            print(f"    {fname:<14} mean={v.mean():8.4f}  std={v.std():7.4f}  "
                  f"min={v.min():8.4f}  max={v.max():8.4f}")

    # Consistency check: consecutive crops of the same clip should be similar.
    print("\n== consistency (same-clip feature stability) ==")
    clips = sorted(set(m["clip"] for m in meta))
    for c in clips:
        sel = np.array([m["clip"] == c for m in meta])
        if sel.sum() < 3:
            continue
        v = X[sel, 0]                      # fg_frac
        print(f"  {c:<24} fg_frac mean={v.mean():.4f} std={v.std():.4f} "
              f"(n={int(sel.sum())})")

    # Weak behavioural check: do features separate real-hand crops from
    # clutter crops?  ('unlabeled' crops come from real-hand clips.)
    hand = labels == "unlabeled"
    clutter = labels == "no-hand"
    if hand.sum() and clutter.sum():
        print("\n== weak check: real-hand crops vs clutter crops ==")
        for j, fname in enumerate(FEATURE_NAMES):
            h, c = X[hand, j], X[clutter, j]
            pooled = np.sqrt((h.var() + c.var()) / 2.0) + 1e-9
            d = (h.mean() - c.mean()) / pooled
            print(f"  {fname:<14} hand={h.mean():8.4f}  clutter={c.mean():8.4f}  "
                  f"cohen_d={d:+6.2f}")
        print("\n  NOTE: 'unlabeled' = real-hand clips, 'no-hand' = clutter clips.")
        print("  This is a mechanics/sanity check, NOT an evaluation: there are no")
        print("  open/fist labels, the sample is tiny and clips leak between splits.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="baseline.pt")
    ap.add_argument("--labels-json", help="{'positive': [...], 'negative': [...]}")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke(args.dataset if os.path.isdir(args.dataset) else "dataset_standin")
        return

    pos = neg = None
    if args.labels_json:
        d = json.load(open(args.labels_json))
        pos, neg = set(d["positive"]), set(d["negative"])
    train_from_directory(args.dataset, args.out, pos, neg)


if __name__ == "__main__":
    main()
