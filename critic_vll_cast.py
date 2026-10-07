"""Critic check: is the retrained verifier robust to *other* artificial light, or
has it just learned this one clip's colour temperature?

The augmentation matched each crop's per-channel mean/std to VeryLowLight's
measured statistics (R/B = 1.79 vs 1.05-1.21 for every other clip).  If the model
only works at that exact cast, it will fall apart when the target is shifted.

So: take the HELD-OUT VeryLowLight crops (never trained on) and re-light them with
a range of cast/brightness shifts, then measure how many hand crops survive and
how much clutter is rejected, for both the previous and the retrained model.

Usage:
    .venv-blazepalm\\Scripts\\python.exe critic_vll_cast.py
"""
import csv
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crop_verifier import SmallCNN, SIZE          # noqa: E402

THRESHOLD = 0.30
DATASET = "vll_dataset"


def relight(img, r_gain, b_gain, bright):
    """Scale each channel and overall brightness: a different lamp."""
    out = img.astype(np.float32)
    out[:, :, 2] *= r_gain        # R
    out[:, :, 0] *= b_gain        # B
    out *= bright
    return np.clip(out, 0, 255).astype(np.uint8)


def load_holdout():
    rows = list(csv.DictReader(open(os.path.join(DATASET, "manifest.csv"))))
    xs, ys = [], []
    for r in rows:
        if r["held_out"] != "1" or not r["source"].startswith("vll_real"):
            continue
        img = cv2.imread(os.path.join(DATASET, r["crop"]))
        if img is None:
            continue
        xs.append(img)
        ys.append(1 if r["label"] == "hand" else 0)
    return xs, np.array(ys)


def model(path):
    m = SmallCNN()
    m.load_state_dict(torch.load(os.path.join(HERE, path), map_location="cpu",
                                 weights_only=True))
    m.eval()
    return m


def score(m, imgs):
    batch = np.stack([cv2.cvtColor(cv2.resize(i, (SIZE, SIZE),
                                              interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB) for i in imgs])
    with torch.no_grad():
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        return torch.sigmoid(m(x)).numpy()


def main():
    imgs, y = load_holdout()
    print(f"held-out VeryLowLight crops: {len(imgs)} "
          f"({int(y.sum())} hand / {int((1-y).sum())} clutter)\n")
    old, new = model("verifier_cnn.pt"), model("verifier_cnn_vll.pt")

    conds = [("as recorded (R/B 1.79)", 1.0, 1.0, 1.0),
             ("warmer   (R/B x2.0)", 1.4, 0.7, 1.0),
             ("cooler   (R/B x0.8)", 0.8, 1.0, 1.0),
             ("near-neutral (R/B x0.6)", 0.6, 1.0, 1.0),
             ("brighter (x1.6)", 1.0, 1.0, 1.6),
             ("darker   (x0.6)", 1.0, 1.0, 0.6),
             ("cooler+brighter", 0.8, 1.0, 1.6)]
    print(f"   {'condition':<26}{'prev kept':>10}{'new kept':>10}"
          f"{'prev cut':>10}{'new cut':>10}")
    for name, rg, bg, br in conds:
        rel = [relight(i, rg, bg, br) for i in imgs]
        po, pn = score(old, rel), score(new, rel)
        print(f"   {name:<26}"
              f"{100*(po[y==1]>=THRESHOLD).mean():>9.1f}%"
              f"{100*(pn[y==1]>=THRESHOLD).mean():>9.1f}%"
              f"{100*(po[y==0]<THRESHOLD).mean():>9.1f}%"
              f"{100*(pn[y==0]<THRESHOLD).mean():>9.1f}%")
    print("\n   'kept' = hand crops the verifier accepts, 'cut' = clutter it rejects.")


if __name__ == "__main__":
    main()
