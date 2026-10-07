"""Item 5: the UNMODIFIED detector's localisation on the same frames the fine-tuned
seeds were scored on (VeryLowLight test blocks, split seed 0).

It imports load_frames() and containment() from finetune_detector_seeds.py rather than
re-implementing them, so the baseline and the fine-tuned numbers use one definition and
one frame list.

    .venv-blazepalm\\Scripts\\python.exe detector_baseline_containment.py

Writes results/detector_baseline_containment.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import finetune_detector_seeds as F                                  # noqa: E402
from blazepalm import PalmDetector                                   # noqa: E402

ML = getattr(F, "ML", os.path.join(HERE, "blazpalm(githubmediapipe)", "BlazePalm", "ML"))


def load_untouched():
    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    return det


def main():
    gt, lm_all, pools = F.load_frames()
    n_train, n_val, n_test = (len(pools["train"]), len(pools["val"]),
                              len(pools["test"]))
    lines = []
    lines.append("Detector containment on the VeryLowLight split (split seed 0)")
    lines.append(f"  frames with a hand: train {n_train}, val {n_val}, test {n_test}")
    lines.append("  containment = a decoded box covers the palm centre, tol 0.15 of the "
                 "box width (containment() in finetune_detector_seeds.py)")

    det = load_untouched()
    for role in ("train", "val", "test"):
        cont, fired = F.containment(det, gt, pools[role])
        lines.append(f"  UNMODIFIED detector, {role:<5} ({len(pools[role])} frames): "
                     f"containment {100*cont:5.1f}%   fired {100*fired:5.1f}%")

    # the fine-tuned equivalents, read from the per-seed records
    import glob
    import json
    js = [json.load(open(p, encoding="utf-8")) for p in
          sorted(glob.glob(os.path.join(HERE, "results", "detector_seeds", "*.json")))]
    if js:
        for key, label in (("test_contain", "test"), ("val_contain", "val")):
            v = np.array([r[key] for r in js], dtype=float)
            lines.append(f"  FINE-TUNED  detector, {label:<5} ({len(v)} seeds): "
                         f"containment {100*v.mean():5.1f}%  SD {100*v.std(ddof=1):4.1f}  "
                         f"range {100*v.min():.1f} to {100*v.max():.1f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "results", "detector_baseline_containment.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
