"""Per-clip F1 for every system, including the CLAHE and gamma MediaPipe variants.

    .venv-blazepalm\\Scripts\\python.exe clip_table.py

Used to check the per-clip sentences in the report prose against the measured file.
"""
import csv
import os

from report_tables import HAND, NOHAND, LABELS, load, RES

rows = load(os.path.join(RES, "summary_all_systems.csv"))
print(f"{'clip':<18}" + "".join(f"{LABELS[k]:>22}" for k in LABELS))
for clip in HAND + NOHAND:
    line = f"{clip:<18}"
    for k in LABELS:
        r = next((x for x in rows if x["system"] == k and x["clip"] == clip), None)
        if r is None:
            line += f"{'-':>22}"
        elif clip in NOHAND:
            line += f"{100*float(r['acceptance_rate']):>21.1f}%"
        else:
            line += f"{100*float(r['f1']):>21.1f}%"
    print(line)
print("\n(nohand clips show the false-lock rate, not F1)")
