"""Report figures from the measured results, sized for a single IEEE column.

    .venv\\Scripts\\python.exe plots_report.py

Writes to results/report_figures/:
  fig_convergence_verifier.png   train/val loss per epoch, mean and SD across seeds
  fig_convergence_detector.png   the same for the detector fine-tune
  fig_confusion_verifier.png     verifier decisions at the chosen threshold, target condition
  fig_seed_auc.png               held-out AUC per seed, boxplot per variant
  fig_seed_f1.png                end-to-end F1 per seed, when the per-seed pipeline
                                 evaluation has been run (seeds_pipeline_eval.py --phase eval)

Every figure is drawn at column width (3.4 in) and 600 dpi, so the font sizes are the
sizes that appear in print. Drawing at double width and scaling down to one column would
halve every label.
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "report_figures")
COL = 3.4                 # IEEE conference column width in inches
DPI = 600

VARIANTS = ["verifier_cnn.pt", "verifier_cnn_big.pt", "verifier_cnn_vll.pt",
            "verifier_dark.pt"]
PRETTY = {
    "verifier_cnn.pt": "11k crops",
    "verifier_cnn_big.pt": "130k crops",
    "verifier_cnn_vll.pt": "target-cond.",
    "verifier_dark.pt": "84k dark-aug.",
    "detector_seed": "detector",
}


def read_curve(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = np.array([int(r["epoch"]) for r in rows])
    tr = np.array([float(r["train_loss_mean"]) for r in rows])
    trs = np.array([float(r["train_loss_sd"]) for r in rows])
    va = np.array([float(r["val_loss_mean"]) for r in rows])
    vas = np.array([float(r["val_loss_sd"]) for r in rows])
    return ep, tr, trs, va, vas


def one_curve_panel(ax, ep, tr, trs, va, vas, title):
    ax.plot(ep, tr, color="C0", label="train")
    ax.fill_between(ep, tr - trs, tr + trs, color="C0", alpha=0.2)
    ax.plot(ep, va, color="C3", label="val")
    ax.fill_between(ep, va - vas, va + vas, color="C3", alpha=0.2)
    if title:
        ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("epoch", fontsize=6)
    ax.set_ylabel("loss", fontsize=6)
    ax.legend(fontsize=5, frameon=False)


def fig_convergence_verifier():
    files = {v: os.path.join(RES, f"verifier_seeds_curves_{v}.csv") for v in VARIANTS}
    have = {v: p for v, p in files.items() if os.path.isfile(p)}
    if not have:
        print("  no verifier curve files yet")
        return
    n = len(have)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(COL, 1.55 * rows), dpi=DPI,
                             sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, (v, path) in zip(axes, have.items()):
        ep, tr, trs, va, vas = read_curve(path)
        one_curve_panel(ax, ep, tr, trs, va, vas, PRETTY.get(v, v))
    for ax in axes[len(have):]:
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, "fig_convergence_verifier.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_convergence_detector():
    path = os.path.join(RES, "detector_seeds_curves.csv")
    if not os.path.isfile(path):
        print("  no detector curve file yet")
        return
    ep, tr, trs, va, vas = read_curve(path)
    fig, ax = plt.subplots(figsize=(COL, 2.1), dpi=DPI)
    one_curve_panel(ax, ep, tr, trs, va, vas, "")
    fig.tight_layout(pad=0.2)
    p = os.path.join(OUT, "fig_convergence_detector.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_confusion():
    """Verifier decisions at the chosen threshold on the target condition, mean of seeds.

    Only the VeryLowLight held-out crops are shown: the per-seed records store HaGRID as an
    AUC rather than as counts, so a HaGRID confusion matrix cannot be built from them
    without re-running prediction, and the target condition is the one that matters here.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, "verifier_seeds", "*_seed*.json"))):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        print("  no per-seed verifier json yet")
        return
    cm = np.zeros((2, 2))
    n = 0
    for r in rows:
        tp, fp, fn, tn = (r.get("test_tp"), r.get("test_fp"), r.get("test_fn"),
                          r.get("test_tn"))
        if None in (tp, fp, fn, tn) or tp == "" or fp == "":
            continue
        cm += np.array([[tp, fn], [fp, tn]], dtype=float)
        n += 1
    if n == 0:
        print("  no confusion counts in the per-seed records yet")
        return
    cm /= n
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(COL, 1.75), dpi=DPI)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({cm[i, j]:.0f})", ha="center",
                    va="center", fontsize=5.5,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1], ["call hand", "call clutter"], fontsize=6)
    ax.set_yticks([0, 1], ["is hand", "is clutter"], fontsize=6)
    ax.set_title(f"Target-condition held-out crops, mean of {n} seeds", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    p = os.path.join(OUT, "fig_confusion_verifier.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_seed_auc():
    """Held-out AUC per seed. HaGRID test crops for the variants trained on them."""
    series, labels = [], []
    for v in VARIANTS:
        vals = []
        for path in sorted(glob.glob(os.path.join(RES, "verifier_seeds",
                                                  f"{v}_seed*.json"))):
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            a = r.get("hag_test_auc")
            if a in (None, "", "nan"):
                a = r.get("vll_test_auc")
            try:
                a = float(a)
            except (TypeError, ValueError):
                continue
            if a == a:
                vals.append(a)
        if vals:
            series.append(np.array(vals))
            labels.append(PRETTY.get(v, v))
    if not series:
        print("  no per-seed AUC values yet")
        return
    fig, ax = plt.subplots(figsize=(COL, 2.0), dpi=DPI)
    ax.boxplot(series, labels=labels, widths=0.55,
               medianprops=dict(color="C3"), flierprops=dict(markersize=2))
    for i, v in enumerate(series, start=1):
        ax.plot(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, "o",
                color="C0", markersize=2, alpha=0.7)
    ax.set_ylabel("held-out AUC", fontsize=6.5)
    ax.set_ylim(0.5, 1.02)
    ax.tick_params(labelsize=5.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(pad=0.2)
    p = os.path.join(OUT, "fig_seed_auc.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_seed_f1():
    series, labels = [], []
    for path in sorted(glob.glob(os.path.join(RES, "seed_eval_*_aggregate.csv"))):
        name = os.path.basename(path).replace("seed_eval_", "").replace("_aggregate.csv", "")
        with open(path, newline="", encoding="utf-8") as f:
            v = np.array([float(r["agg_f1"]) for r in csv.DictReader(f)])
        series.append(v)
        labels.append(PRETTY.get(name, name.replace("_", " ")))
    if not series:
        print("  no per-seed aggregate files yet (seeds_pipeline_eval.py --phase eval)")
        return
    fig, ax = plt.subplots(figsize=(COL, 2.0), dpi=DPI)
    ax.boxplot(series, labels=labels, widths=0.55,
               medianprops=dict(color="C3"), flierprops=dict(markersize=2))
    ax.set_ylabel("aggregate F1 (hand clips)", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(pad=0.2)
    p = os.path.join(OUT, "fig_seed_f1.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_convergence_verifier()
    fig_convergence_detector()
    fig_confusion()
    fig_seed_auc()
    fig_seed_f1()
    print(f"\nfigures -> {OUT}")
