"""Hand labelling tool for a subset of the VeryLowLight test frames.

Why this exists: every number in the report is scored against MediaPipe's own static
detector, which makes the yardstick and the baseline the same model family. These labels
are an independent opinion, so the report can show what MediaPipe and the pipeline look
like against a human instead of against MediaPipe.

Which frames: 150 frames drawn from the test blocks of the split in vll_split.py. Test
blocks only, so labelling cannot leak into any model decision, and spread evenly across
the blocks so the subset spans the clip's brightness drift. The frame list is written to
results/human_labels/frames_to_label.txt.

Two ways to use it.

Interactive (needs a display, run it yourself):
    .venv\\Scripts\\python.exe label_frames.py
  Mouse drag draws a box around the hand. Keys:
     n  no hand in this frame
     u  unsure / cannot tell
     space  accept the MediaPipe box drawn on screen (fast path)
     b  back one frame
     q  save and quit  (safe to stop and resume later)

Export mode (no display needed, label the PNGs in any image viewer afterwards):
    .venv\\Scripts\\python.exe label_frames.py --export
  Writes the 150 frames as PNGs plus results/human_labels/template.csv for you to fill in.

Both modes write results/human_labels/human_labels.csv with
    frame,label,x1,y1,x2,y2
where label is hand, no_hand or unsure, and the box is the hand box in clip pixels
(empty for no_hand and unsure). Re-running either mode skips frames already labelled.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vll_split import BLOCK, SPLIT_SEED, block_roles        # noqa: E402

CLIP = "VeryLowLight.mp4"
GT_NPZ = "gt_verylowlight.npz"
GT_KEY = "static_conf0.3"
OUTDIR = os.path.join(HERE, "results", "human_labels")
LABELS_CSV = os.path.join(OUTDIR, "human_labels.csv")
FRAMES_TXT = os.path.join(OUTDIR, "frames_to_label.txt")
PER_BLOCK = 30          # 5 test blocks x 30 = 150 frames


def test_blocks():
    roles = block_roles(SPLIT_SEED)
    return sorted(b for b, r in roles.items() if r == "test")


def select_frames(n_frames=651):
    """Evenly spaced frames inside each test block."""
    blocks = test_blocks()
    frames = []
    for b in blocks:
        lo = b * BLOCK + 1
        hi = min((b + 1) * BLOCK, n_frames)
        if hi < lo:
            continue
        span = hi - lo + 1
        k = min(PER_BLOCK, span)
        picks = np.linspace(lo, hi, k).round().astype(int)
        frames.extend(sorted(set(int(p) for p in picks)))
    return frames


def load_gt():
    z = np.load(os.path.join(HERE, GT_NPZ))
    return z[GT_KEY]


def load_labelled():
    done = {}
    if os.path.isfile(LABELS_CSV):
        for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")):
            done[int(r["frame"])] = r
    return done


def append_label(row):
    new = not os.path.isfile(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "label", "x1", "y1", "x2", "y2"])
        if new:
            w.writeheader()
        w.writerow(row)


def gt_for(gt, n):
    row = gt[gt[:, 0] == n]
    if len(row) and row[0, 1]:
        cx, cy, pw = float(row[0, 2]), float(row[0, 3]), float(row[0, 4])
        return cx, cy, pw
    return None


def export(frames, gt):
    os.makedirs(OUTDIR, exist_ok=True)
    frames_dir = os.path.join(OUTDIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(HERE, CLIP))
    want = set(frames)
    n = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if n in want:
            path = os.path.join(frames_dir, f"vll_frame_{n:04d}.png")
            cv2.imwrite(path, frame)
            written += 1
    cap.release()
    tpl = os.path.join(OUTDIR, "template.csv")
    with open(tpl, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label", "x1", "y1", "x2", "y2"])
        for n in frames:
            w.writerow([n, "", "", "", "", ""])
    print(f"wrote {written} frames to {frames_dir}")
    print(f"wrote {tpl}  (fill in label and, for a hand, the box; then rename it "
          f"to human_labels.csv)")
    return 0


def interactive(frames, gt):
    cap = cv2.VideoCapture(os.path.join(HERE, CLIP))
    images = {}
    want = set(frames)
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if n in want:
            images[n] = frame.copy()
    cap.release()

    done = load_labelled()
    todo = [n for n in frames if n not in done]
    if not todo:
        print("all selected frames already labelled")
        return 0
    print(f"{len(todo)} frames to label of {len(frames)} selected")
    print("drag a box around the hand, or press n (no hand), u (unsure), "
          "space (accept the MediaPipe box), b (back), q (save and quit)")

    state = {"x1": -1, "y1": -1, "x2": -1, "y2": -1, "drag": False}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(x1=x, y1=y, x2=x, y2=y, drag=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state.update(x2=x, y2=y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.update(x2=x, y2=y, drag=False)

    win = "label VeryLowLight frames"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    i = 0
    while 0 <= i < len(todo):
        n = todo[i]
        frame = images[n].copy()
        g = gt_for(gt, n)
        if g is not None:
            cx, cy, pw = g
            half = 1.3 * pw
            cv2.rectangle(frame,
                          (int(cx - half), int(cy - half)),
                          (int(cx + half), int(cy + half)), (0, 200, 255), 1)
            cv2.putText(frame, "MediaPipe box (press space to accept, or draw your own)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            cv2.putText(frame, "MediaPipe sees no hand here",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"frame {n}   {i+1}/{len(todo)}",
                    (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        if state["x2"] > 0:
            cv2.rectangle(frame, (state["x1"], state["y1"]), (state["x2"], state["y2"]),
                          (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            if i > 0:
                i -= 1
                state.update(x1=-1, y1=-1, x2=-1, y2=-1)
            continue
        if key == ord("n"):
            append_label(dict(frame=n, label="no_hand", x1="", y1="", x2="", y2=""))
            i += 1
            state.update(x1=-1, y1=-1, x2=-1, y2=-1)
            continue
        if key == ord("u"):
            append_label(dict(frame=n, label="unsure", x1="", y1="", x2="", y2=""))
            i += 1
            state.update(x1=-1, y1=-1, x2=-1, y2=-1)
            continue
        if key == ord(" "):
            if g is None:
                append_label(dict(frame=n, label="no_hand", x1="", y1="", x2="", y2=""))
            else:
                cx, cy, pw = g
                half = 1.3 * pw
                append_label(dict(frame=n, label="hand", x1=int(cx - half),
                                  y1=int(cy - half), x2=int(cx + half),
                                  y2=int(cy + half)))
            i += 1
            state.update(x1=-1, y1=-1, x2=-1, y2=-1)
            continue
        if key == ord("s") and state["x2"] > state["x1"] and state["y2"] > state["y1"]:
            append_label(dict(frame=n, label="hand", x1=state["x1"], y1=state["y1"],
                              x2=state["x2"], y2=state["y2"]))
            i += 1
            state.update(x1=-1, y1=-1, x2=-1, y2=-1)

    cv2.destroyAllWindows()
    done = load_labelled()
    print(f"labelled {len(done)} of {len(frames)} selected frames -> {LABELS_CSV}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true",
                    help="write PNGs and a template CSV instead of opening a window")
    ap.add_argument("--list", action="store_true", help="just print the frame list")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    frames = select_frames()
    gt = load_gt()
    with open(FRAMES_TXT, "w", encoding="utf-8") as f:
        f.write(f"# VeryLowLight test-block frames to label (split seed {SPLIT_SEED})\n")
        f.write(f"# {len(frames)} frames, blocks {test_blocks()}\n")
        f.write("\n".join(str(n) for n in frames) + "\n")
    print(f"{len(frames)} frames selected from test blocks {test_blocks()}")
    print(f"list -> {FRAMES_TXT}")
    if args.list:
        print(" ".join(str(n) for n in frames))
        return 0
    if args.export:
        return export(frames, gt)
    return interactive(frames, gt)


if __name__ == "__main__":
    sys.exit(main())
