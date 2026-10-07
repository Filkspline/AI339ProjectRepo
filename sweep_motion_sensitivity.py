"""Step 1.1: which lever limits sensitivity to SLOW hand motion?

The motion-blob path works on fast motion but not on slow/subtle movement.  Three
candidate causes, tested independently against real footage:

  * MOTION_PIXEL_THR (12/255 per-pixel change)
  * MOTION_RING (5 frames, ~170 ms at 30 fps -- for the 15 fps clips, ~330 ms)
  * MOTION_BLOB_MIN_SIDE_FRAC (0.03 x frame width blob-size floor)

All three are swept in ONE pass per clip (a 9-frame ring lets every ring length be
evaluated from the same frames, and one difference image serves every pixel
threshold), so the comparison is on identical data.

For each combination it reports, per clip:
    blob rate       - fraction of frames with any blob above the size floor
    on-hand rate    - fraction of frames whose LARGEST blob is on the hand
                      (MediaPipe truth where available)
and the noise cost is the same blob rate on the no-hand clips, where every blob is
a false proposal.

Usage:
    .venv-blazepalm\\Scripts\\python.exe sweep_motion_sensitivity.py
"""
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

CLIPS = [("CloseLightMoving.mp4", "gt_clips.npz"),
         ("CloseLightStill.mp4", "gt_clips.npz"),
         ("CloseDarkStill.mp4", "gt_clips.npz"),
         ("NohandLight.mp4", "gt_clips.npz"),
         ("NohandDark.mp4", "gt_clips.npz"),
         ("VeryLowLight.mp4", "gt_verylowlight.npz")]
WIDTH = 320
RING = 9                       # long ring; sub-windows give the shorter options
KS = [2, 4, 6, 8]              # frames spanned: ring length 3, 5, 7, 9
THRS = [3.0, 6.0, 12.0]
FLOORS = [0.02, 0.03, 0.05]
GT_KEYS = {"VeryLowLight.mp4": "static_conf0.3"}


def hand_at(gt, n, window=5):
    row = gt[gt[:, 0] == n]
    if len(row) and row[0, 1]:
        return float(row[0, 2]), float(row[0, 3])
    present = gt[gt[:, 1] == 1][:, 0]
    if len(present):
        near = present[np.abs(present - n) <= window]
        if len(near):
            k = near[np.argmin(np.abs(near - n))]
            row = gt[gt[:, 0] == k]
            return float(row[0, 2]), float(row[0, 3])
    return None


def largest_blob(mask, width, floors):
    """Largest connected component's centroid plus its per-floor pass flags.

    The floor is applied to AREA, exactly as MotionGate.blobs() does
    (min_area = (frac * width)^2).  Filtering on the bbox side instead is a
    different and much more permissive test: a long thin streak of changed
    pixels passes a side test but is not a hand-sized blob, which is what made
    the first version of this sweep report noise blobs in a static room.
    """
    n_lab, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lab <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, w, h, area = stats[i]
    return (float(cents[i][0]), float(cents[i][1]), area / float(width * width),
            max(w, h) / float(width),
            {f: area >= (f * width) ** 2 for f in floors})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=250)
    args = ap.parse_args()

    # combo -> per-clip counters
    combos = [(k, t) for k in KS for t in THRS]
    stats = {c: {} for c in combos}
    for clip, _g in CLIPS:
        z = np.load(os.path.join(HERE, _g))
        gt = z[GT_KEYS.get(clip, clip)]
        cap = cv2.VideoCapture(os.path.join(HERE, clip))
        ring = deque(maxlen=RING)
        n = 0
        per = {c: dict(frames=0, handed=0, blob={f: 0 for f in FLOORS},
                       onhand={f: 0 for f in FLOORS}, onsample=0)
               for c in combos}
        while True:
            ret, frame = cap.read()
            if not ret or n >= args.frames:
                break
            n += 1
            fh, fw = frame.shape[:2]
            scale = WIDTH / float(fw)
            small = cv2.resize(frame, (WIDTH, max(2, int(round(fh * scale)))),
                               interpolation=cv2.INTER_AREA)
            lu = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            ring.append(lu)
            if len(ring) < RING:
                continue
            h = hand_at(gt, n)
            # MediaPipe hand in ring coordinates
            hand_rc = None
            if h is not None:
                hand_rc = (h[0] * scale, h[1] * scale)
            newest = ring[-1]
            for k in KS:
                old = ring[-1 - k]
                # frame-wide gain compensation, exactly as MotionGate does
                g = float(newest.std() / (old.std() + 1e-6))
                old_c = g * old + (float(newest.mean()) - g * float(old.mean()))
                a = newest - newest.mean()
                b = old_c - old_c.mean()
                d = np.abs(a - b)
                for t in THRS:
                    c = (k, t)
                    per[c]["frames"] += 1
                    if hand_rc is not None:
                        per[c]["handed"] += 1
                    lb = largest_blob((d > t).astype(np.uint8), WIDTH, FLOORS)
                    if lb is None:
                        continue
                    cx, cy, _areafrac, _side, passes = lb
                    for f in FLOORS:
                        if passes[f]:
                            per[c]["blob"][f] += 1
                            if hand_rc is not None:
                                if (float(np.hypot(cx - hand_rc[0], cy - hand_rc[1]))
                                        / WIDTH <= 0.12):
                                    per[c]["onhand"][f] += 1
        cap.release()
        stats_key = clip
        for c in combos:
            stats[c][stats_key] = per[c]

    print(f"analysis width {WIDTH}, ring up to {RING} frames, "
          f"clips up to {args.frames} frames each")
    print("blob rate = % of frames with the largest component above the size"
          " floor; on-hand = % of frames where that blob is on the MediaPipe hand\n")
    for f in FLOORS:
        print(f"== blob-size floor {f:.2f} x frame width ==")
        hdr = f"   {'span':>5}{'thr':>6}"
        for clip, _ in CLIPS:
            hdr += f"{clip.split('.')[0][:11]:>13}"
        print(hdr)
        for k in KS:
            for t in THRS:
                c = (k, t)
                line = f"   {k+1:>5}{t:>6.0f}"
                for clip, _ in CLIPS:
                    per = stats[c][clip]
                    nf = max(per["frames"], 1)
                    if clip.startswith("Nohand"):
                        v = 100.0 * per["blob"][f] / nf
                        line += f"{v:>12.1f}%"
                    else:
                        nh = max(per["handed"], 1)
                        v = 100.0 * per["onhand"][f] / nh
                        line += f"{v:>12.1f}%"
                print(line)
        print()

    print("columns: Close clips + VeryLowLight show ON-HAND % (higher = more"
          " sensitive);")
    print("         Nohand clips show FALSE blob rate % (higher = more noise).")


if __name__ == "__main__":
    main()
