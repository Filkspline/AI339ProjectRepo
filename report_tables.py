"""Print the report's tables straight from the measured files, so every number in the
report can be checked against its source.

    .venv-blazepalm\\Scripts\\python.exe report_tables.py

Nothing here computes a new measurement: it aggregates the per-clip rows that
evaluate_all_systems.py wrote, and reads the multi-seed and statistics files if they
exist.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

HAND = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
        "FarLight.mp4", "FarDark.mp4", "VeryLowLight.mp4"]
NOHAND = ["NohandLight.mp4", "NohandDark.mp4"]
LABELS = {
    "mp_raw": "MediaPipe (raw)",
    "mp_clahe": "MediaPipe + CLAHE",
    "mp_gamma": "MediaPipe + gamma 0.5",
    "ours_default": "Ours default",
    "ours_big61": "Ours big verifier",
    "dark_v3": "Dark v3",
    "dark_v4": "Dark v4",
}
CLIP_SHORT = {"CloseLightMoving.mp4": "CloseLightMoving",
              "CloseLightStill.mp4": "CloseLightStill",
              "CloseDarkStill.mp4": "CloseDarkStill",
              "FarLight.mp4": "FarLight", "FarDark.mp4": "FarDark",
              "VeryLowLight.mp4": "VeryLowLight",
              "NohandLight.mp4": "NohandLight", "NohandDark.mp4": "NohandDark",
              "HandheldStill.mp4": "HandheldStill"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text_any(path):
    """PowerShell '>' redirects write UTF-16; the Python-written files are UTF-8."""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def aggregate(rows, clips, key):
    rs = [r for r in rows if r["clip"] in clips and r["system"] == key]
    f = sum(int(r["frames"]) for r in rs)
    t = sum(int(r["truth_frames"]) for r in rs)
    a = sum(int(r["accepted"]) for r in rs)
    tp = sum(int(r["tp"]) for r in rs)
    fp = sum(int(r["fp"]) for r in rs)
    prec = tp / a if a else float("nan")
    rec = tp / t if t else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if a and t and (prec + rec) else float("nan")
    return dict(frames=f, truth=t, accepted=a, tp=tp, fp=fp, precision=prec,
                recall=rec, f1=f1, acc=a / f if f else float("nan"))


def table1():
    print("=" * 96)
    print("TABLE 1  main comparison, one session, identical truth")
    print("=" * 96)
    rows = load(os.path.join(RES, "summary_all_systems.csv"))
    print(f"{'system':<22}{'frames':>7}{'truth':>6}{'acc':>6}{'acc%':>7}"
          f"{'prec%':>7}{'rec%':>7}{'F1%':>7}{'locks%':>8}")
    for k, lab in LABELS.items():
        a = aggregate(rows, HAND, k)
        nh = aggregate(rows, NOHAND, k)
        print(f"{lab:<22}{a['frames']:>7}{a['truth']:>6}{a['accepted']:>6}"
              f"{100*a['acc']:>7.1f}{100*a['precision']:>7.1f}{100*a['recall']:>7.1f}"
              f"{100*a['f1']:>7.1f}{100*nh['acc']:>8.1f}")


def table2():
    print()
    print("=" * 96)
    print("TABLE 2  per-clip F1 (%), same session")
    print("=" * 96)
    rows = load(os.path.join(RES, "summary_all_systems.csv"))
    keys = ["mp_raw", "ours_default", "ours_big61", "dark_v3", "dark_v4"]
    head = f"{'clip':<18}" + "".join(f"{LABELS[k]:>18}" for k in keys)
    print(head)
    for clip in HAND:
        line = f"{CLIP_SHORT[clip]:<18}"
        for k in keys:
            r = next(x for x in rows if x["system"] == k and x["clip"] == clip)
            line += f"{float(r['f1'])*100:>18.1f}"
        print(line)
    for clip in NOHAND:
        line = f"{CLIP_SHORT[clip]:<18}"
        for k in keys:
            r = next(x for x in rows if x["system"] == k and x["clip"] == clip)
            line += f"{float(r['acceptance_rate'])*100:>18.1f}"
        print(line + "   (false-lock rate)")


def table3():
    print()
    print("=" * 96)
    print("TABLE 3  VeryLowLight by split role (per-frame)")
    print("=" * 96)
    path = os.path.join(RES, "summary_vll_roles.csv")
    if not os.path.isfile(path):
        print("missing summary_vll_roles.csv; run evaluate_all_systems.py")
        return
    rows = load(path)
    print(f"{'system':<22}{'role':<7}{'frames':>7}{'truth':>7}{'acc':>6}{'acc%':>7}"
          f"{'prec%':>7}{'rec%':>7}{'F1%':>7}")
    for k, lab in LABELS.items():
        for role in ("train", "val", "test"):
            r = next((x for x in rows if x["system"] == k and x["role"] == role), None)
            if r is None:
                continue
            print(f"{lab:<22}{role:<7}{r['frames']:>7}{r['truth_frames']:>7}"
                  f"{r['accepted']:>6}{100*float(r['acceptance_rate']):>7.1f}"
                  f"{100*float(r['precision']):>7.1f}{100*float(r['recall']):>7.1f}"
                  f"{100*float(r['f1']):>7.1f}")


def table_seeds():
    print()
    print("=" * 96)
    print("TABLE 4  multi-seed results (mean and SD over seeds)")
    print("=" * 96)
    # Per-seed JSONs are written as each seed finishes; the summary CSV is only
    # written when the whole run ends, so prefer the JSONs and fall back to the CSV.
    by = {}
    for p in sorted(glob.glob(os.path.join(RES, "verifier_seeds", "*_seed*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        by.setdefault(r["variant"], []).append(r)
    source = "results/verifier_seeds/*.json"
    path = os.path.join(RES, "verifier_seeds_summary.csv")
    if not by and os.path.isfile(path):
        for r in load(path):
            by.setdefault(r["variant"], []).append(r)
        source = "verifier_seeds_summary.csv"
    if by:
        fields = ["val_auc", "hag_test_auc", "vll_val_auc", "vll_test_auc",
                  "test_f1", "test_precision", "test_recall", "threshold",
                  "best_epoch", "train_seconds", "params"]
        print(f"source: {source}")
        print(f"{'variant':<26}{'dataset':<20}{'seeds':>6}"
              + "".join(f"{f:>14}" for f in fields))
        for v, rs in by.items():
            line = f"{v:<26}{rs[0]['dataset']:<20}{len(rs):>6}"
            for f in fields:
                vals = np.array([float(r[f]) for r in rs
                                 if r.get(f, "") not in ("", None)], dtype=float)
                vals = vals[~np.isnan(vals)]
                line += (f"{vals.mean():>8.4f}+-{vals.std(ddof=1):<5.4f}"
                         if len(vals) > 1 else f"{'n/a':>14}")
            print(line)
    else:
        print("no verifier seed results yet")

    path = os.path.join(RES, "detector_seeds_summary.csv")
    if os.path.isfile(path):
        rows = load(path)
        fields = ["test_contain", "val_contain", "light_mean_contain", "best_epoch",
                  "train_seconds", "collapse_hot_anchors"]
        print(f"\n{'detector fine-tune':<24}{'seeds':>6}"
              + "".join(f"{f:>20}" for f in fields))
        line = f"{'palmdetector_dark.pth':<24}{len(rows):>6}"
        for f in fields:
            v = np.array([float(r[f]) for r in rows], dtype=float)
            line += f"{v.mean():>12.4f}+-{v.std(ddof=1):<7.4f}"
        print(line)
        trips = sum(1 for r in rows if r["collapse_tripped"] == "True")
        print(f"  collapse guard tripped for {trips} of {len(rows)} seeds")
    else:
        print("\nmissing detector_seeds_summary.csv")


def table_stats():
    print()
    print("=" * 96)
    print("TABLE 5  paired tests (McNemar exact, block bootstrap 50-frame, Holm-Bonferroni)")
    print("=" * 96)
    path = os.path.join(RES, "stats_paired_frames.csv")
    if os.path.isfile(path):
        rows = [r for r in load(path) if r.get("clips", "hand") == "hand"]
        print("scope: six hand clips (the same aggregate as Table 1); the CSV also has")
        print("rows for all eight clips, where 'correct' means accepting nothing")
        print(f"{'A':<14}{'B':<14}{'A only':>7}{'B only':>7}{'p':>11}{'p_holm':>11}"
              f"{'dF1':>8}{'95% CI':>22}{'sig':>6}")
        for r in rows:
            ci = (f"[{float(r['boot_lo']):+.4f},{float(r['boot_hi']):+.4f}]")
            print(f"{r['a']:<14}{r['b']:<14}{r['b_only_a_correct']:>7}"
                  f"{r['c_only_b_correct']:>7}{float(r['p_mcnemar']):>11.2e}"
                  f"{float(r['p_holm']):>11.2e}{float(r['f1_diff']):>8.4f}"
                  f"{ci:>22}{r['significant']:>6}")
    else:
        print("missing stats_paired_frames.csv")
    path = os.path.join(RES, "stats_across_seeds.csv")
    if os.path.isfile(path):
        rows = load(path)
        print(f"\n{'A':<26}{'B':<26}{'test':<10}{'n':>4}{'p':>10}{'p_holm':>10}"
              f"{'effect':>9}{'mean dF1':>10}")
        for r in rows:
            print(f"{r['a']:<26}{r['b']:<26}{r['test']:<10}{r['n']:>4}"
                  f"{float(r['p']):>10.4f}{float(r['p_holm']):>10.4f}"
                  f"{float(r['effect']):>9.3f}{float(r['mean_diff']):>10.4f}")
    else:
        print("\nmissing stats_across_seeds.csv")


def attribution():
    print()
    print("=" * 96)
    print("DETECTOR ATTRIBUTION (from the funnel logs in build/)")
    print("=" * 96)
    for name in ("funnel_new_default.log", "funnel_new_big61.log"):
        p = os.path.join(HERE, "build", name)
        if not os.path.isfile(p):
            continue
        print(f"--- {name} ---")
        for line in read_text_any(p).splitlines():
            if any(k in line for k in ("frames with a hand", "detector did not fire",
                                       "detector fired, but", "size gate rejected",
                                       "motion gate rejected", "CNN verifier rejected",
                                       "survived the whole chain")):
                print("  " + line.rstrip())


class Tee:
    """Mirror stdout into results/report_tables.txt so the report has a snapshot."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


if __name__ == "__main__":
    tee = Tee(os.path.join(RES, "report_tables.txt"))
    sys.stdout = tee
    table1()
    table2()
    table3()
    table_seeds()
    table_stats()
    attribution()
    tee.flush()
    sys.stdout = tee.out
    tee.f.close()
    print("written:", os.path.join(RES, "report_tables.txt"))
