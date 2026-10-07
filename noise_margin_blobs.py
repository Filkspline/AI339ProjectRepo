"""Step 1.2: noise margin at the new BLOB threshold.

The blob proposal path now runs at 6/255 per pixel instead of 12/255, which moves
it closer to the sensor-noise floor.  The original motion-gate work found the
static-room/held-still separation collapsing at sigma ~ 8; this re-measures that
margin for the blob path specifically, at both thresholds, so the new operating
point is not assumed safe.

For each injected noise level it reports:
    static-room false blobs  (NohandLight/NohandDark - every blob is a false
                              proposal, so this is the noise cost)
    held-still on-hand blobs (CloseLightStill - the slow-motion case we are
                              trying to recover)

Usage:
    .venv-blazepalm\\Scripts\\python.exe noise_margin_blobs.py
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

import hand_mouse_cursor as hmc                             # noqa: E402

WIDTH = 320
RING = 5
SIGMAS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]


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


def run(clip, gt, thr, floor, sigma, frames, seed=7):
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    ring = deque(maxlen=RING)
    n = 0
    fired = 0
    onhand = 0
    handed = 0
    while True:
        ret, frame = cap.read()
        if not ret or n >= frames:
            break
        n += 1
        if sigma:
            frame = np.clip(frame.astype(np.float32)
                            + rng.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)
        fh, fw = frame.shape[:2]
        scale = WIDTH / float(fw)
        small = cv2.resize(frame, (WIDTH, max(2, int(round(fh * scale)))),
                           interpolation=cv2.INTER_AREA)
        lu = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ring.append(lu)
        if len(ring) < RING:
            continue
        new, old = ring[-1], ring[0]
        g = float(new.std() / (old.std() + 1e-6))
        old = g * old + (float(new.mean()) - g * float(old.mean()))
        d = np.abs((new - new.mean()) - (old - old.mean()))
        mask = (d > thr).astype(np.uint8)
        n_lab, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h = hand_at(gt, n)
        if h is not None:
            handed += 1
        if n_lab <= 1:
            continue
        i = 1 + int(np.argmax(stats[1:, 4]))
        area = stats[i, 4]
        if area < (floor * WIDTH) ** 2:
            continue
        fired += 1
        if h is not None:
            cx, cy = float(cents[i][0]), float(cents[i][1])
            if float(np.hypot(cx - h[0] * scale, cy - h[1] * scale)) / WIDTH <= 0.12:
                onhand += 1
    cap.release()
    return n, fired, onhand, handed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    z = np.load(os.path.join(HERE, "gt_clips.npz"))

    configs = [("gate thr 12, area 0.03 (shipped gate)", 12.0, 0.03),
               ("blob thr 6,  area 0.03", 6.0, 0.03),
               ("blob thr 6,  area 0.05 (NEW)", 6.0, 0.05),
               ("blob thr 12, area 0.05", 12.0, 0.05)]
    print(f"analysis width {WIDTH}, ring {RING}, Gaussian noise added per frame"
          f"\n(clips: CloseLightStill = held-still hand, Nohand* = static room)\n")
    for tag, thr, floor in configs:
        print(f"== {tag} ==")
        print(f"   {'sigma':>6}{'CloseLightStill on-hand':>26}"
              f"{'NohandLight false blobs':>26}{'NohandDark false blobs':>25}")
        for sigma in SIGMAS:
            n1, f1, oh, handed = run("CloseLightStill.mp4", z["CloseLightStill.mp4"],
                                     thr, floor, sigma, args.frames)
            n2, f2, _o, _h = run("NohandLight.mp4", z["NohandLight.mp4"],
                                 thr, floor, sigma, args.frames)
            n3, f3, _o, _h = run("NohandDark.mp4", z["NohandDark.mp4"],
                                 thr, floor, sigma, args.frames)
            print(f"   {sigma:>6.1f}{100*oh/max(handed,1):>25.1f}%"
                  f"{100*f2/max(n2,1):>25.1f}%{100*f3/max(n3,1):>24.1f}%")
        print()


if __name__ == "__main__":
    main()
