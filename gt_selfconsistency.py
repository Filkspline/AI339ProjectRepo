"""How much does the yardstick itself move on the frames picked for hand labelling?

The 150 frames in results/human_labels/frames_to_label.txt are the ones a person is
being asked to label, because every system in the report is currently scored against
MediaPipe rather than against a human. This script measures what the two available
MediaPipe truths (static mode, the evaluation yardstick, and tracking mode, what the
baseline system reports) say about those same 150 frames, so the circularity of the
ground truth is quantified rather than asserted.

    .venv-blazepalm\\Scripts\\python.exe gt_selfconsistency.py

Writes results/human_labels/gt_selfconsistency.txt
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "human_labels")
GT = os.path.join(HERE, "gt_verylowlight.npz")
FRAMES = os.path.join(OUT, "frames_to_label.txt")


def load_frames():
    out = []
    for line in open(FRAMES, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(int(line))
    return out


def centre(gt, frame):
    """(present, cx, cy, palm width) for a 1-based frame index, or None."""
    row = gt[frame - 1]
    present = bool(row[1])
    return present, float(row[2]), float(row[3]), float(row[4])


def main():
    z = np.load(GT)
    static = z["static_conf0.3"]
    track = z["conf0.3"]
    frames = load_frames()
    lines = []
    lines.append(f"frames selected for hand labelling: {len(frames)}")
    lines.append(f"clip VeryLowLight, {static.shape[0]} frames total, "
                 f"split seed 0 test blocks [1, 6, 8, 9, 12]")

    s_hand = [f for f in frames if centre(static, f)[0]]
    t_hand = [f for f in frames if centre(track, f)[0]]
    both = [f for f in frames if centre(static, f)[0] and centre(track, f)[0]]
    neither = [f for f in frames if not centre(static, f)[0] and not centre(track, f)[0]]
    s_only = [f for f in frames if centre(static, f)[0] and not centre(track, f)[0]]
    t_only = [f for f in frames if not centre(static, f)[0] and centre(track, f)[0]]

    lines.append("")
    lines.append("what each truth says about the same 150 frames")
    lines.append(f"  static MediaPipe sees a hand : {len(s_hand):>3} "
                 f"({100*len(s_hand)/len(frames):.1f}%)")
    lines.append(f"  tracking MediaPipe sees one  : {len(t_hand):>3} "
                 f"({100*len(t_hand)/len(frames):.1f}%)")
    lines.append(f"  both agree a hand is present : {len(both):>3}")
    lines.append(f"  both agree no hand           : {len(neither):>3}")
    lines.append(f"  static only                  : {len(s_only):>3}")
    lines.append(f"  tracking only                : {len(t_only):>3}")
    agree = len(both) + len(neither)
    lines.append(f"  agreement on presence        : {agree}/{len(frames)} "
                 f"= {100*agree/len(frames):.1f}%")

    if both:
        # frame width of the clip, so the offset can be quoted the same way the
        # match radius is
        import cv2
        cap = cv2.VideoCapture(os.path.join(HERE, "VeryLowLight.mp4"))
        ok, fr = cap.read()
        fw = fr.shape[1] if ok else 0
        cap.release()
        offs = []
        for f in both:
            _, sx, sy, sw = centre(static, f)
            _, tx, ty, tw = centre(track, f)
            offs.append(np.hypot(sx - tx, sy - ty) / fw if fw else np.nan)
        offs = np.array(offs, dtype=float)
        lines.append("")
        lines.append("when both see the hand, how far apart are the two positions?")
        lines.append(f"  frame width {fw} px; offset in frame widths")
        lines.append(f"  median {np.median(offs):.4f}  p90 {np.percentile(offs, 90):.4f}  "
                     f"max {offs.max():.4f}")
        # the match radius used by the evaluation is max(0.06 frame widths,
        # 0.6 x palm width); count how often the two truths are further apart
        # than that, which is the same tolerance a scored position gets
        rad = np.array([max(0.06, 0.6 * centre(static, f)[3] / fw) for f in both])
        beyond = int(np.sum(offs > rad))
        lines.append(f"  further apart than the evaluation's own match radius: "
                     f"{beyond} of {len(both)} ({100*beyond/len(both):.1f}%)")

    txt = "\n".join(lines)
    print(txt)
    path = os.path.join(OUT, "gt_selfconsistency.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
