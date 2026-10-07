"""Filter the mined HaGRID crops with an independent model before training.

build_big_dataset.py mines positives with the palm detector on gesture-labelled
images -- but "this image contains a gesture" does not mean "this detection is on
the hand", and the VeryLowLight work showed the detector fires confidently while
off the hand.  Training on those crops would teach the verifier to accept the
detector's own false positives as hands, which is what a permissive boundary looks
like afterwards.

So every mined crop is re-checked with a DIFFERENT network -- the landmark model's
presence head ("is this crop a hand?") -- and only crops it confirms are kept.
Measured on our daylight clips this head scores 1.000 on real hands and ~0.03 on
clutter, so it is a usable arbiter for HaGRID's daylight images.  It is NOT usable
in the VeryLowLight condition (presence ~0.000 there even on correct crops), which
is why the VeryLowLight crops keep the MediaPipe arbiter they were mined with.

Reports the pass rate, which is a direct estimate of how often the detector's
top detection on a gesture image is NOT the hand.

Usage:
    .venv-blazepalm\\Scripts\\python.exe filter_mined.py --threshold 0.5
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from handlandmarks import HandLandmarks                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="big_dataset")
    ap.add_argument("--mined-csv", default=None,
                    help="manifest of mined crops (default: <dataset>/mined.csv)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="filtered manifest (default: <dataset>/mined_filtered.csv)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    ds = os.path.join(HERE, args.dataset)
    mined_csv = args.mined_csv or os.path.join(ds, "mined.csv")
    out_csv = args.out or os.path.join(ds, "mined_filtered.csv")
    rows = list(csv.DictReader(open(mined_csv)))
    print(f"{len(rows)} mined crops from {mined_csv}")

    model = HandLandmarks()
    model.load_weights(os.path.join(ML, "HandLandmarks.pth"))
    model.eval()

    scores = []
    kept = []
    bs = args.batch
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        batch = []
        for r in chunk:
            img = cv2.imread(os.path.join(ds, r["crop"]))
            if img is None:
                batch.append(None)
                continue
            # the landmark model wants its native 256; crops are stored at 128
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            batch.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        valid = [b for b in batch if b is not None]
        if not valid:
            continue
        x = torch.from_numpy(np.stack(valid)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.no_grad():
            presence, _handed, _reg = model(x)
        presence = presence.reshape(-1).tolist()
        it = iter(presence)
        for r, b in zip(chunk, batch):
            if b is None:
                continue
            p = float(next(it))
            scores.append(p)
            if p >= args.threshold:
                kept.append(r)
        if (i // bs) % 20 == 0:
            print(f"   scored {len(scores)}/{len(rows)}", flush=True)

    s = np.array(scores)
    print(f"\nlandmark-model presence on the mined crops (n={len(s)}):")
    for q in (10, 25, 50, 75, 90):
        print(f"   p{q:<3} {np.percentile(s, q):.3f}")
    for t in (0.3, 0.5, 0.7, 0.9):
        print(f"   presence >= {t}: {100*np.mean(s >= t):5.1f}% kept")
    print(f"\n-> decides the filter at {args.threshold}: keeping {len(kept)} "
          f"of {len(s)} ({100*len(kept)/max(len(s),1):.1f}%)")
    print("   i.e. an estimated "
          f"{100*(1-len(kept)/max(len(s),1)):.1f}% of the detector's top "
          "detections on gesture images were NOT confirmed as hand crops.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"wrote {len(kept)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
