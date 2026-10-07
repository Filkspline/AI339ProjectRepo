"""Structural check on the report source: environments balanced, labels resolvable,
figures present on disk. Not a substitute for compiling, but it catches the mistakes
that a compile would turn into an error.

    .venv\\Scripts\\python.exe check_report_tex.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "report", "AIML339_Final_Report.tex")

src = open(TEX, encoding="utf-8").read()
bad = 0
for env in ("table", "tabular", "figure", "itemize", "thebibliography", "abstract",
            "document", "IEEEkeywords"):
    b = len(re.findall(r"\\begin\{" + env + r"\}", src))
    e = len(re.findall(r"\\end\{" + env + r"\}", src))
    flag = "ok" if b == e else "MISMATCH"
    bad += b != e
    print(f"  {env:<16} begin {b}  end {e}  {flag}")

delta = src.count("{") - src.count("}")
print(f"  brace delta      {delta}  {'ok' if delta == 0 else 'MISMATCH'}")
bad += delta != 0

labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
missing = sorted(refs - labels)
print(f"  labels {sorted(labels)}")
print(f"  refs   {sorted(refs)}")
print(f"  unresolved refs: {missing if missing else 'none'}")
bad += bool(missing)

for fig in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src):
    p = os.path.join(HERE, "report", "figures", fig)
    exists = os.path.isfile(p)
    print(f"  figure {fig:<34} {'present' if exists else 'MISSING ' + p}")
    bad += not exists

cites = set(re.findall(r"\\cite\{([^}]+)\}", src))
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", src))
print(f"  citations {sorted(cites)}")
print(f"  bibitems  {sorted(bib)}")
print(f"  cited but not in bibliography: {sorted(cites - bib) or 'none'}")

print(f"\n  words: {len(re.findall(r'[A-Za-z][A-Za-z-]*', src))}")
print("  RESULT:", "clean" if not bad else f"{bad} problem(s)")
