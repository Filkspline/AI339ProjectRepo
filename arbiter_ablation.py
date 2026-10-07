"""Arbitered versus unfiltered verifier, measured on one identical set of held-out crops.

The report used to compare two models trained in an earlier round (held-out AUC 0.8081
unfiltered against 0.9650 filtered). Those numbers came from the old two-way split, so the
comparison is re-measured here on the HaGRID test crops of the current source-based split:
the same crops, the same loader, the same AUC code for every checkpoint.

    .venv-blazepalm\\Scripts\\python.exe arbiter_ablation.py

Writes results/arbiter_ablation.txt
"""
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                                      # noqa: E402

import train_verifier_seeds as T                                  # noqa: E402
from crop_verifier import SIZE, SmallCNN                          # noqa: E402
from train_verifier_seeds import auc, predict, role_of_row        # noqa: E402


def make_verifier(path):
    m = SmallCNN()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

DATASET = "big_dataset"
CHECKPOINTS = [
    ("unfiltered (earlier round)", "verifier_cnn_big_UNFILTERED.pt"),
    ("arbitered, released", "verifier_cnn_big.pt"),
    ("arbitered, current run seed 0", os.path.join("results", "verifier_seeds",
                                                   "verifier_cnn_big.pt_seed0.pt")),
]


def load_test_crops():
    path = os.path.join(HERE, DATASET, "manifest.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    X, y, seen = [], [], {}
    for r in rows:
        role = role_of_row(r)
        if role != "test":
            continue
        img = cv2.imread(os.path.join(HERE, DATASET, r["crop"]))
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y.append(1 if r["label"] == "hand" else 0)
        seen[r["source"]] = seen.get(r["source"], 0) + 1
    return np.stack(X).astype(np.uint8), np.array(y, dtype=np.float32), seen


def main():
    X, y, seen = load_test_crops()
    lines = [f"dataset {DATASET}, held-out role 'test' only",
             f"  sources: {seen}",
             f"  crops: {len(y)}  ({int((y == 1).sum())} hand, {int((y == 0).sum())} clutter)",
             ""]
    for label, rel in CHECKPOINTS:
        p = os.path.join(HERE, rel)
        if not os.path.isfile(p):
            lines.append(f"{label:<32} missing: {rel}")
            continue
        model = make_verifier(p)
        n = sum(pp.numel() for pp in model.parameters())
        s = predict(model, X)
        lines.append(f"{label:<32} AUC {auc(s, y):.4f}  params {n}  ({rel})")
        lines.append(f"{'':<32} mean P(hand) on hand crops {s[y == 1].mean():.4f}, "
                     f"on clutter crops {s[y == 0].mean():.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "results", "arbiter_ablation.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
