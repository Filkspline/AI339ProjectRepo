"""Generate the LaTeX for the report tables that come from the multi-seed and
statistics runs, straight from the results files. Nothing is typed by hand, so a
table cannot drift from its source.

    .venv-blazepalm\\Scripts\\python.exe make_latex_tables.py

Writes results/latex_fragments.tex and prints the same text. Copy the fragments
into report/AIML339_Final_Report.tex.
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

VARIANT_LABEL = {
    "verifier_cnn.pt": "11k mined",
    "verifier_cnn_big.pt": "130k arbitered",
    "verifier_cnn_vll.pt": "32k VLL",
    "verifier_dark.pt": "84k dark",
    "verifier_dark_v4.pt": "dark v4",
    "detector_seed": "detector fine-tune",
}
ORDER = ["verifier_cnn.pt", "verifier_cnn_big.pt", "verifier_cnn_vll.pt",
         "verifier_dark.pt", "verifier_dark_v4.pt"]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ms(vals, fmt="{:.3f}"):
    """mean +- SD over seeds, or a dash when there is only one value."""
    v = np.array([float(x) for x in vals if x not in ("", None)], dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return "--"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())} $\\pm$ {fmt.format(v.std(ddof=1))}"


def verifier_rows():
    by = {}
    for p in sorted(glob.glob(os.path.join(RES, "verifier_seeds", "*_seed*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        by.setdefault(r["variant"], []).append(r)
    return by


def pipeline_rows():
    """Seed-level pipeline aggregates from seeds_pipeline_eval.py."""
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, "seed_eval_*_aggregate.csv"))):
        stem = os.path.basename(p).replace("seed_eval_", "").replace("_aggregate.csv", "")
        out[stem] = load(p)
    return out


def tab_seeds():
    by = verifier_rows()
    pipe = pipeline_rows()
    names = [n for n in ORDER if n in by]
    if not names:
        return "% no verifier seed results yet"
    lines = [
        r"\begin{table}[t]",
        r"\caption{Verifier variants retrained from scratch with the shared seed list, mean "
        r"and standard deviation over the seeds that completed. `HaGRID' is held-out AUC on "
        r"the HaGRID test crops (blank where the variant's pool contains none), `target' is "
        r"held-out AUC on the VeryLowLight test blocks. The seed campaign was stopped "
        r"before it finished; the seed counts are in the table.",
        r"\label{tab:seeds}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Variant & HaGRID & VLL & F1\% & Prec.\% & Rec.\% & Locks\% & Thr. & Epochs \\",
        r"\midrule",
    ]
    for n in names:
        rs = by[n]
        agg = pipe.get(n, [])
        label = VARIANT_LABEL.get(n, n)
        hac = ms([r.get("hag_test_auc", "nan") for r in rs], "{:.3f}")
        vla = ms([r.get("vll_test_auc", "nan") for r in rs], "{:.3f}")
        f1 = ms([r["agg_f1"] for r in agg], "{:.1f}") if agg else "--"
        pr = ms([float(r["agg_precision"]) * 100 for r in agg], "{:.1f}") if agg else "--"
        rc = ms([float(r["agg_recall"]) * 100 for r in agg], "{:.1f}") if agg else "--"
        lk = ms([float(r["false_lock_rate"]) * 100 for r in agg], "{:.1f}") if agg else "--"
        thr = ms([r["threshold"] for r in rs], "{:.2f}")
        ep = ms([r["best_epoch"] for r in rs], "{:.0f}")
        lines.append(f"{label} & {hac} & {vla} & {f1} & {pr} & {rc} & {lk} & {thr} & {ep} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tab_detector():
    pipe = pipeline_rows()
    agg = pipe.get("detector_seed")
    if not agg:
        return "% no detector seed pipeline results yet"
    by = {}
    for p in sorted(glob.glob(os.path.join(RES, "detector_seeds", "detector_seed*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        by.setdefault("detector_seed", []).append(r)
    rs = by.get("detector_seed", [])
    trips = sum(1 for r in rs if r.get("collapse_tripped"))
    lines = [
        r"\begin{table}[t]",
        r"\caption{Detector fine-tune (six prediction heads, 2.49\% of parameters) "
        r"retrained on ten seeds with the shared seed list, mean and standard deviation "
        r"over seeds. Containment is the share of target-condition test frames where a "
        r"detection box contains the hand. The last three columns are the whole dark "
        r"pipeline on the eight clips with that seed's detector.}",
        r"\label{tab:detector}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & Contain\% & Val contain\% & Train s & F1\% & Prec.\% & Rec.\% \\",
        r"\midrule",
        f"Detector fine-tune & "
        f"{ms([r.get('test_contain', 'nan') for r in rs], '{:.1f}')} & "
        f"{ms([r.get('val_contain', 'nan') for r in rs], '{:.1f}')} & "
        f"{ms([r.get('train_seconds', 'nan') for r in rs], '{:.0f}')} & "
        f"{ms([r['agg_f1'] for r in agg], '{:.1f}')} & "
        f"{ms([float(r['agg_precision']) * 100 for r in agg], '{:.1f}')} & "
        f"{ms([float(r['agg_recall']) * 100 for r in agg], '{:.1f}')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        f"% collapse guard tripped for {trips} of {len(rs)} seeds",
    ]
    return "\n".join(lines)


SHORT = {
    "mp_raw": "MP raw", "mp_clahe": "MP CLAHE", "mp_gamma": "MP gamma",
    "ours_default": "default", "ours_big61": "big", "dark_v3": "dark v3",
    "dark_v4": "dark v4",
}


def tab_stats(only=None):
    path = os.path.join(RES, "stats_paired_frames.csv")
    if not os.path.isfile(path):
        return "% no statistics yet"
    rows = [r for r in load(path) if r.get("clips", "hand") == "hand"]
    if only:
        rows = [r for r in rows if (r["a"], r["b"]) in only]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Paired tests on per-frame decisions over the six hand clips "
        r"(1553 frames). `A/B' counts frames where A made the correct decision and B "
        r"did not, and the reverse; $p$ is the exact McNemar test on those discordant "
        r"pairs, Holm-Bonferroni adjusted across all ten comparisons. $\Delta$F1 is "
        r"accompanied by a 95\% block-bootstrap interval over 50-frame blocks (2000 "
        r"resamples, seed 12345). A dagger marks a frame-level difference that is "
        r"significant while the F1 interval still spans zero.}",
        r"\label{tab:stats}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"A & B & A/B & $p$ (Holm) & $\Delta$F1 & 95\% CI & Sig. \\",
        r"\midrule",
    ]
    for r in rows:
        a = SHORT.get(r["a"], r["a"])
        b = SHORT.get(r["b"], r["b"])
        p = float(r["p_holm"])
        ptxt = f"{p:.0e}" if p < 0.01 else f"{p:.3f}"
        lo, hi = float(r["boot_lo"]), float(r["boot_hi"])
        sig = "yes" if r["significant"] == "True" else "no"
        # a dagger marks a frame-level difference that is significant while the
        # F1 interval still contains zero: the two tests answer different questions
        dagger = r"$^{\dagger}$" if (sig == "yes" and lo <= 0 <= hi) else ""
        lines.append(f"{a} & {b} & {r['b_only_a_correct']}/{r['c_only_b_correct']} & "
                     f"{ptxt} & {float(r['f1_diff']):+.2f} & "
                     f"{lo:+.2f},{hi:+.2f} & {sig}{dagger} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"% $\dagger$ significant frame-level difference, F1 interval spans zero",
    ]
    return "\n".join(lines)


def main():
    frags = {
        "tab:seeds": tab_seeds(),
        "tab:detector": tab_detector(),
        "tab:stats": tab_stats(),
    }
    out = []
    for k, v in frags.items():
        out.append(f"%% ================= {k} =================")
        out.append(v)
        out.append("")
        print(f"%% ================= {k} =================")
        print(v)
        print()
    path = os.path.join(RES, "latex_fragments.tex")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print("written:", path)


if __name__ == "__main__":
    main()
