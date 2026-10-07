"""Find an operating threshold for the blob-specific verifier.

blob_verifier.py showed a large separation gain (+37..+43 points vs -18..-34 for
the main verifier), but the ABSOLUTE off-hand acceptance is what decides whether
the pipeline locks onto moving clutter, and at threshold 0.30 it is still 47-55%.
This sweeps the threshold on held-out blob groups to see whether an operating
point exists that keeps clutter acceptance low while retaining useful hand recall.

Usage:
    .venv-blazepalm\\Scripts\\python.exe sweep_blob_threshold.py
"""
import csv
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crop_verifier import SmallCNN                     # noqa: E402
from blob_verifier import load, score_all, group_by_blob  # noqa: E402


def main():
    X, y, meta = load(os.path.join(HERE, "blob_dataset"))
    ho = np.array([m["held_out"] == "1" for m in meta])
    models = {}
    for name, path in (("blob_verifier.pt", "blob_verifier.pt"),
                       ("verifier_cnn.pt", "verifier_cnn.pt"),
                       ("verifier_cnn_vll.pt", "verifier_cnn_vll.pt")):
        p = os.path.join(HERE, path)
        if not os.path.isfile(p):
            continue
        m = SmallCNN()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        m.eval()
        models[name] = group_by_blob(meta, score_all(m, X))

    print("held-out blob groups only; 'on' = blob on the MediaPipe hand,")
    print("'off' = blob with no MediaPipe hand within 0.20 frame widths\n")
    for name, rows in models.items():
        rows = [r for r in rows if r["held_out"] == "1"]
        on = {k: np.array([r[k] for r in rows if r["label"] == "hand"])
              for k in ("mx", "mean", "single")}
        off = {k: np.array([r[k] for r in rows if r["label"] == "clutter"])
               for k in ("mx", "mean", "single")}
        print(f"== {name} (n on={len(on['mx'])}, off={len(off['mx'])}) ==")
        print(f"   {'thr':>6}" + "".join(f"{k + ' on/off':>18}"
                                         for k in ("mx", "mean", "single")))
        for thr in (0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99):
            line = f"   {thr:>6.2f}"
            for k in ("mx", "mean", "single"):
                a = 100 * float((on[k] >= thr).mean())
                b = 100 * float((off[k] >= thr).mean())
                line += f"{a:>9.1f}/{b:<7.1f}%"
            print(line)
        print()


if __name__ == "__main__":
    main()
