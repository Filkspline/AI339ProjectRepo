"""Export per-clip F1 for the six systems in the report's figure, from the one-session
evaluation, as a CSV the bar chart can be checked against.

    .venv-blazepalm\\Scripts\\python.exe export_per_clip_f1.py

Writes results/per_clip_f1.csv (F1 in percent, one row per clip, plus the no-hand clips
which carry the false-lock rate instead and are marked in the kind column).
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# report column name -> system key in summary_all_systems.csv
COLUMNS = [("MP", "mp_raw"), ("MP+CLAHE", "mp_clahe"), ("Default", "ours_default"),
           ("Big", "ours_big61"), ("Dark v3", "dark_v3"), ("Dark v4", "dark_v4")]
CLIP_ORDER = ["CloseLightMoving", "CloseLightStill", "CloseDarkStill", "FarLight",
              "FarDark", "VeryLowLight", "NohandLight", "NohandDark"]


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "summary_all_systems.csv"),
                                    newline="", encoding="utf-8")))
    by = {(r["system"], r["clip"].replace(".mp4", "")): r for r in rows}
    out = []
    for clip in CLIP_ORDER:
        subset = [r for (s, c), r in by.items() if c == clip]
        if not subset:
            continue
        kind = subset[0]["kind"]
        rec = {"clip": clip, "kind": kind,
               "frames": subset[0]["frames"], "truth_frames": subset[0]["truth_frames"],
               "metric": "f1_percent" if kind == "hand" else "false_lock_rate_percent"}
        for label, key in COLUMNS:
            r = by.get((key, clip))
            if r is None:
                rec[label] = ""
            elif kind == "hand":
                rec[label] = round(100 * float(r["f1"]), 1)
            else:
                rec[label] = round(100 * float(r["acceptance_rate"]), 1)
        out.append(rec)

    path = os.path.join(RES, "per_clip_f1.csv")
    fields = ["clip", "kind", "metric", "frames", "truth_frames"] + [c[0] for c in COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}\n")
    print(f"{'clip':<18}{'kind':<8}" + "".join(f"{c[0]:>11}" for c in COLUMNS))
    for r in out:
        print(f"{r['clip']:<18}{r['kind']:<8}"
              + "".join(f"{r[c[0]]:>11}" for c in COLUMNS))
    print("\nNohandLight/NohandDark rows are the false-lock rate, not F1 "
          "(see the metric column).")


if __name__ == "__main__":
    main()
