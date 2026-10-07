"""Statistics for the report, on per-frame outcomes and on per-seed results.

Three families of tests.

1. Paired per-frame tests (McNemar, exact binomial). Two systems answer the same frame,
   so the informative cells are the discordant ones. A frame counts as correct if the
   system accepted a position within the match radius of a real hand, or if there was no
   hand and it accepted nothing. That makes "did nothing on an empty frame" correct,
   which the raw correct column in the per-frame files does not, since a rejection there
   is recorded as incorrect.

2. Block bootstrap confidence intervals for F1 differences. Fifty-frame blocks are
   resampled with replacement inside each clip, so the temporal correlation within a block
   is preserved and both systems see the same resampled frames. Fixed seed, recorded.

3. Paired across-seed tests on the trained variants. Ten seeds per variant, so the pairs
   are matched by seed. Shapiro-Wilk decides between a paired t-test and Wilcoxon
   signed-rank; the effect size is Cohen's d for the t-test and the matched-pairs
   rank-biserial correlation for Wilcoxon.

Holm-Bonferroni is applied within each family. Alpha is 0.05.

    .venv\\Scripts\\python.exe stats_tests.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PF = os.path.join(RES, "per_frame")
BOOT_SEED = 12345
BOOT_N = 2000
BLOCK = 50
ALPHA = 0.05

SYSTEMS = ["mp_raw", "mp_clahe", "mp_gamma", "ours_default", "ours_big61",
           "dark_v3", "dark_v4"]

PAIRS = [
    ("mp_raw", "ours_default"),
    ("mp_raw", "ours_big61"),
    ("mp_raw", "dark_v3"),
    ("mp_raw", "dark_v4"),
    ("mp_raw", "mp_clahe"),
    ("mp_raw", "mp_gamma"),
    ("mp_clahe", "mp_gamma"),
    ("ours_default", "ours_big61"),
    ("ours_big61", "dark_v3"),
    ("dark_v3", "dark_v4"),
]


def load_per_frame():
    """{system: {clip: [(frame, truth, accepted, correct, dist)]}}"""
    out = {}
    for s in SYSTEMS:
        path = os.path.join(PF, f"per_frame_{s}.csv")
        if not os.path.isfile(path):
            path = os.path.join(RES, f"per_frame_{s}.csv")
        if not os.path.isfile(path):
            continue
        per_clip = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                per_clip.setdefault(r["clip"], []).append(
                    (int(r["frame"]), r["truth_present"] == "True",
                     r["accepted"] == "True", r["correct"] == "True"))
        out[s] = per_clip
    return out


def frame_correct(truth, accepted, matched_flag):
    """Decision accuracy for one frame: hand found, or correctly nothing."""
    if truth:
        return bool(accepted and matched_flag)
    return not bool(accepted)


def mcnemar(a_correct, b_correct):
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = st.binomtest(b, n, 0.5).pvalue
    return b, c, float(p)


def f1_of(correct, truth, accepted):
    """F1 in the project's convention, so it matches summary_all_systems.csv:
    precision is over accepted frames (a frame accepted in the wrong place is a
    false positive), recall is over truth frames. The two conventions differ for
    frames that are accepted but off the hand, so mixing them would put two
    different F1 values for the same system in one report."""
    tp = int(np.sum(correct & accepted))
    fp = int(np.sum(accepted & ~correct))
    fn = int(np.sum(truth & ~correct))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    if prec == prec and rec == rec and (prec + rec) > 0:
        return 2 * prec * rec / (prec + rec)
    return float("nan")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def block_bootstrap(pf, sa, sb, n=BOOT_N, seed=BOOT_SEED, keep=None):
    """CI for F1(a) - F1(b), resampling 50-frame blocks inside each clip."""
    rng = np.random.default_rng(seed)
    clips = sorted((set(pf[sa]) & set(pf[sb])) & keep) if keep else \
        sorted(set(pf[sa]) & set(pf[sb]))
    blocks = {}
    for clip in clips:
        ra = {r[0]: r for r in pf[sa][clip]}
        rb = {r[0]: r for r in pf[sb][clip]}
        frames = sorted(set(ra) & set(rb))
        bl = []
        for i in range(0, len(frames), BLOCK):
            chunk = frames[i:i + BLOCK]
            bl.append(np.array([(ra[f][1], ra[f][2], ra[f][3],
                                 rb[f][2], rb[f][3]) for f in chunk], dtype=bool))
        blocks[clip] = bl
    diffs = []
    for _ in range(n):
        pick = {c: [blocks[c][i] for i in rng.integers(0, len(blocks[c]),
                                                       len(blocks[c]))]
                for c in clips}
        arr = np.concatenate([np.concatenate(pick[c]) for c in clips])
        truth, acc_a, ok_a, acc_b, ok_b = (arr[:, 0], arr[:, 1], arr[:, 2],
                                           arr[:, 3], arr[:, 4])
        cor_a = np.where(truth, acc_a & ok_a, ~acc_a)
        cor_b = np.where(truth, acc_b & ok_b, ~acc_b)
        fa = f1_of(cor_a, truth, acc_a)
        fb = f1_of(cor_b, truth, acc_b)
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    return (float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5)),
            float(np.nanmean(diffs)))


def run_pairs(pf, scope, clips):
    """One family of paired tests, restricted to `clips`. Returns a list of rows."""
    print(f"== McNemar, paired per-frame decision accuracy ({scope}) ==")
    tests = []
    for sa, sb in PAIRS:
        if sa not in pf or sb not in pf:
            continue
        cor_a, cor_b, truth, acc_a, acc_b = [], [], [], [], []
        for clip in sorted((set(pf[sa]) & set(pf[sb])) & clips):
            ra = {r[0]: r for r in pf[sa][clip]}
            rb = {r[0]: r for r in pf[sb][clip]}
            for f in sorted(set(ra) & set(rb)):
                truth.append(ra[f][1])
                acc_a.append(ra[f][2])
                acc_b.append(rb[f][2])
                cor_a.append(frame_correct(ra[f][1], ra[f][2], ra[f][3]))
                cor_b.append(frame_correct(rb[f][1], rb[f][2], rb[f][3]))
        truth = np.array(truth); acc_a = np.array(acc_a); acc_b = np.array(acc_b)
        cor_a = np.array(cor_a); cor_b = np.array(cor_b)
        b, c, p = mcnemar(cor_a, cor_b)
        f1a = f1_of(cor_a, truth, acc_a)
        f1b = f1_of(cor_b, truth, acc_b)
        lo, hi, mean = block_bootstrap(pf, sa, sb, keep=clips)
        tests.append(dict(clips=scope, a=sa, b=sb, b_only_a_correct=b,
                          c_only_b_correct=c, p_mcnemar=p, f1_a=f1a, f1_b=f1b,
                          f1_diff=f1a - f1b, boot_lo=lo, boot_hi=hi,
                          boot_mean=mean, n_frames=len(truth)))
    adj = holm([t["p_mcnemar"] for t in tests]) if tests else []
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)
        t["significant"] = bool(pa < ALPHA)
        print(f"  {t['a']:<14} vs {t['b']:<14} discordant {t['b_only_a_correct']:>5}/"
              f"{t['c_only_b_correct']:<5} p={t['p_mcnemar']:.2e} "
              f"p_holm={t['p_holm']:.2e} dF1={t['f1_diff']:+.4f} "
              f"[{t['boot_lo']:+.4f},{t['boot_hi']:+.4f}] "
              f"{'sig' if t['significant'] else 'not sig'}")
    return tests


def main():
    pf = load_per_frame()
    missing = [s for s in SYSTEMS if s not in pf]
    if missing:
        print(f"note: no per-frame file for {missing}")
    if len(pf) < 2:
        sys.exit("need at least two systems' per-frame outcomes; run "
                 "evaluate_all_systems.py first")

    # Two scopes. "hand" matches Table 1 of the report (six hand clips, 1553
    # frames); "all" adds the two no-hand clips, where a correct decision is
    # accepting nothing, so F1 there is not comparable with Table 1.
    hand_clips = {c for s in pf for c in pf[s] if not c.lower().startswith("nohand")}
    all_clips = {c for s in pf for c in pf[s]}
    out = run_pairs(pf, "hand", hand_clips)
    print()
    out += run_pairs(pf, "all", all_clips)
    with open(os.path.join(RES, "stats_paired_frames.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # ------------------------------------------- across-seed paired comparisons
    print("\n== across-seed comparisons (paired by seed) ==")
    seed_rows = []
    for path in sorted(glob.glob(os.path.join(RES, "seed_eval_*_aggregate.csv"))):
        name = os.path.basename(path).replace("seed_eval_", "").replace("_aggregate.csv", "")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        seed_rows.append((name, {int(r["seed"]): float(r["agg_f1"]) for r in rows}))
    seed_tests = []
    for i in range(len(seed_rows)):
        for j in range(i + 1, len(seed_rows)):
            na, va = seed_rows[i]
            nb, vb = seed_rows[j]
            common = sorted(set(va) & set(vb))
            if len(common) < 5:
                continue
            a = np.array([va[s] for s in common])
            b = np.array([vb[s] for s in common])
            d = a - b
            if np.allclose(d, 0):
                seed_tests.append(dict(a=na, b=nb, n=len(common), test="none",
                                       statistic=float("nan"), p=1.0,
                                       effect=0.0, mean_diff=0.0))
                continue
            w_stat, w_p = st.wilcoxon(a, b)
            sw_p = st.shapiro(d).pvalue if len(d) >= 3 else 0.0
            normal = sw_p > 0.05
            if normal:
                t_stat, t_p = st.ttest_rel(a, b)
                eff = float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) else float("nan")
                seed_tests.append(dict(a=na, b=nb, n=len(common), test="paired t",
                                       statistic=float(t_stat), p=float(t_p),
                                       effect=eff, mean_diff=float(np.mean(d)),
                                       shapiro_p=float(sw_p)))
            else:
                ranks = st.rankdata(np.abs(d))
                rp = ranks[d > 0].sum(); rn = ranks[d < 0].sum()
                eff = float((rp - rn) / (rp + rn)) if (rp + rn) else 0.0
                seed_tests.append(dict(a=na, b=nb, n=len(common), test="wilcoxon",
                                       statistic=float(w_stat), p=float(w_p),
                                       effect=eff, mean_diff=float(np.mean(d)),
                                       shapiro_p=float(sw_p)))
    if seed_tests:
        adj = holm([t["p"] for t in seed_tests])
        for t, pa in zip(seed_tests, adj):
            t["p_holm"] = float(pa)
            t["significant"] = bool(pa < ALPHA)
            print(f"  {t['a']:<28} vs {t['b']:<28} {t['test']:<10} n={t['n']} "
                  f"p={t['p']:.4f} p_holm={t['p_holm']:.4f} effect={t['effect']:+.3f} "
                  f"mean dF1={t['mean_diff']:+.4f} "
                  f"{'sig' if t['significant'] else 'not sig'}")
        with open(os.path.join(RES, "stats_across_seeds.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(seed_tests[0].keys()))
            w.writeheader()
            w.writerows(seed_tests)

    meta = dict(alpha=ALPHA, bootstrap_seed=BOOT_SEED, bootstrap_n=BOOT_N,
                bootstrap_block=BLOCK,
                note=("Holm-Bonferroni applied within each family. McNemar is the exact "
                      "binomial test on discordant pairs. A frame is correct if a hand "
                      "present was found within the match radius, or if no hand was "
                      "present and nothing was accepted."))
    with open(os.path.join(RES, "stats_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote stats_paired_frames.csv, stats_across_seeds.csv, stats_meta.json")


if __name__ == "__main__":
    main()
