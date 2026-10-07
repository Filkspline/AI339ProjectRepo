"""Page budget for the report, measured from the source instead of guessed.

There is no LaTeX engine and no network on this machine, so the document cannot be
compiled here. This estimates the printed length from the same quantities LaTeX would
consume: body words, float heights from the real image aspect ratios, and table heights
from row counts and wrapping in the p{} columns.

    .venv\\Scripts\\python.exe page_budget.py

Assumptions, all stated in the output:
  * IEEEtran conference, A4: text width 7.07 in, two columns, column height 9.25 in.
  * Body text at 10 pt two-column: 55 words per column-inch (about 1,020 words/page).
  * \\footnotesize (8 pt) line height 0.135 in, average glyph width 0.068 in.
  * Floats are packed without waste; a real run leaves 5-15% of each column empty.
"""
import os
import re
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "report", "AIML339_Final_Report.tex")
FIGDIR = os.path.join(HERE, "report", "figures")

TEXT_WIDTH = 7.07          # inches, IEEEtran conference A4
COLUMN_HEIGHT = 9.25       # inches of usable column
WORDS_PER_COL_INCH = 55.0
FOOT_ROW_IN = 0.135        # footnotesize line height
CHAR_IN = 0.068            # footnotesize average glyph width
CAPTION_ROW_IN = 0.125
RULE_IN = 0.06


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def words_of(s):
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", s))


def table_height(tbl):
    cap = re.search(r"\\caption\{(.+?)\}\s*\\label", tbl, re.S)
    cap_lines = 0
    if cap:
        cap_lines = max(1, -(-words_of(cap.group(1)) // 16))   # ~16 words per caption line
    tab = re.search(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", tbl, re.S)
    if not tab:
        return cap_lines * CAPTION_ROW_IN, 0
    inner = tab.group(1)
    spec, _, body = inner.partition("\n")
    widths = []
    for m in re.finditer(r"p\{([\d.]+)\\linewidth\}", spec):
        widths.append(float(m.group(1)) * 3.4)                 # column widths in inches
    fixed = spec.count("c") + spec.count("l") + spec.count("r")
    rows, lines_total = 0, 0
    for raw in re.split(r"\\\\", body):
        cells = [c.strip() for c in raw.split("&")]
        cells = [c for c in cells if c and "rule" not in c and "addlinespace" not in c]
        if not cells:
            continue
        rows += 1
        visual = 1
        for i, cell in enumerate(cells):
            if i < len(widths) and widths[i] > 0:
                chars = max(1, int(widths[i] / CHAR_IN))
                visual = max(visual, -(-words_of(cell) * 6 // chars))   # words -> chars
        lines_total += visual
    h = lines_total * FOOT_ROW_IN + cap_lines * CAPTION_ROW_IN + RULE_IN * 2
    return h, rows


def figure_height(fig):
    inc = re.search(r"\\includegraphics\[width=([\d.]+)\\textwidth\]\{([^}]+)\}", fig)
    cap = re.search(r"\\caption\{(.+?)\}\s*\\label", fig, re.S)
    cap_lines = max(1, -(-words_of(cap.group(1)) // 16)) if cap else 0
    if not inc:
        return 0.0, None
    frac, name = float(inc.group(1)), inc.group(2)
    path = os.path.join(FIGDIR, name)
    size = png_size(path) if os.path.isfile(path) else None
    height = 0.0
    if size:
        w_in = frac * TEXT_WIDTH
        height = w_in * size[1] / size[0]
    return height + cap_lines * CAPTION_ROW_IN, (name, size, frac * TEXT_WIDTH, height)


def main():
    src = open(TEX, encoding="utf-8").read()
    src = re.sub(r"(?m)^\s*%.*$", "", src)
    body = src.split(r"\begin{document}")[1]
    main = body.split(r"\section*{Statement}")[0]
    tail = body[len(main):]

    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", main, re.S)
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", main, re.S)
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", main, flags=re.S)
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", text, flags=re.S)

    words = words_of(text)
    refs_words = words_of(tail.split(r"\begin{thebibliography}")[0])
    bib_words = words_of(tail.split(r"\begin{thebibliography}")[-1])

    print(f"main content: {words} words, {len(tables)} tables, {len(figures)} figures")
    print(f"back matter:  Statement {refs_words} words, References {bib_words} words\n")

    col_in = words / WORDS_PER_COL_INCH
    print(f"body text      {words:>5} words -> {col_in:6.2f} column-inches")

    print("\ntables")
    tf = 0.0
    for t in tables:
        h, rows = table_height(t)
        label = re.search(r"\\label\{([^}]+)\}", t)
        print(f"  {label.group(1) if label else '?':<14} {rows:>2} rows -> {h:5.2f} in tall")
        tf += h

    print("\nfigures")
    ff = 0.0
    for f in figures:
        h, info = figure_height(f)
        label = re.search(r"\\label\{([^}]+)\}", f)
        if info:
            name, size, w_in, imh = info
            state = f"{size[0]}x{size[1]} px" if size else "FILE MISSING"
            print(f"  {label.group(1) if label else '?':<14} {name:<24} {state:<16} "
                  f"width {w_in:.2f} in -> {imh:.2f} in image + caption = {h:.2f} in")
        else:
            print(f"  {label.group(1) if label else '?':<14} no includegraphics")
        ff += h

    floats = tf + ff
    print(f"\nfloats total   {floats:.2f} column-inches "
          f"({tf:.2f} tables, {ff:.2f} figures)")

    per_page = 2 * COLUMN_HEIGHT
    print(f"\nper page       {per_page:.2f} column-inches (2 columns x {COLUMN_HEIGHT} in)")
    print("\npages, at a range of assumptions:")
    for wpi in (60, 55, 50):                    # words per column-inch
        for waste in (0.0, 0.10, 0.15):
            ci = words / wpi + floats * (1 + waste)
            print(f"  {wpi:>2} words/col-in, floats {int(waste*100):>2}% wasted"
                  f" -> {ci/per_page:5.2f} pages  ({ci:.1f} col-in)")
    print("\nback matter is outside the five-page limit by the rubric, and is not counted")


if __name__ == "__main__":
    main()
