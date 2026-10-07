"""Step 1 check: three repeats of the same configuration must agree exactly.

The ms/frame column is a wall-clock performance measurement, not a decision input, so it
is excluded from the comparison. Everything else, including every accepted-frame count,
precision, recall, F1 and false-lock rate, must match to the last digit.
"""
import csv
import pathlib
import sys

RUNS = [f"build/det_run{i}.csv" for i in (1, 2, 3)]
IGNORE = {"ms_per_frame"}


def load(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["clip"], r["pipeline"])] = {k: v for k, v in r.items()
                                                if k not in IGNORE}
    return rows


def main():
    for p in RUNS:
        if not pathlib.Path(p).is_file():
            print(f"missing {p}")
            return 1
    a, b, c = (load(p) for p in RUNS)
    diff_ab = [k for k in a if a[k] != b.get(k)]
    diff_ac = [k for k in a if a[k] != c.get(k)]
    if not diff_ab and not diff_ac:
        print(f"IDENTICAL: {len(a)} clip/pipeline rows match across all three runs")
        print("(ms_per_frame excluded, it measures wall clock)")
        return 0
    print(f"DIFFERENCES: {len(diff_ab)} rows differ between run 1 and 2, "
          f"{len(diff_ac)} between run 1 and 3")
    for k in sorted(set(diff_ab) | set(diff_ac))[:8]:
        print(f"\n  {k}")
        for field in a[k]:
            if a[k][field] != b.get(k, {}).get(field) or a[k][field] != c.get(k, {}).get(field):
                print(f"    {field}: run1={a[k][field]!r} run2={b.get(k,{}).get(field)!r} "
                      f"run3={c.get(k,{}).get(field)!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
