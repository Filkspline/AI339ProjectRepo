"""MediaPipe per-frame positions under three input conditions, for the baselines.

Runs in the venv that has mediapipe:
    .venv\\Scripts\\python.exe make_mp_positions.py

Conditions, all fed to an unmodified mp.solutions.hands:
  raw     the frame as recorded
  clahe   CLAHE on the L channel in LAB, clipLimit 2.0, 8x8 tiles
  gamma   luminance gamma, gamma 0.5 (brightens), applied through a 256-entry LUT

Modes:
  track   static_image_mode=False, what the MediaPipe baseline column reports
  static  static_image_mode=False -> True equivalently, no temporal state, which is the
          yardstick. Saved for every condition so the yardstick can be checked against
          the raw one, but the report's truth is always the raw static run.

The CLAHE and gamma parameters are the usual textbook values, fixed in advance. They
were not tuned on any part of the data, so no selection happened on the test blocks.

Writes results/mp_positions.npz with one (N,5) array per "<condition>_<mode>_<clip>":
    [frame_index, present, cx_px, cy_px, palm_width_px]
"""
import os
import sys
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe is not installed in this interpreter. Run this with .venv, "
             "not .venv-blazepalm.")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "mp_positions.npz")

CLIPS = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
         "FarLight.mp4", "FarDark.mp4", "NohandLight.mp4", "NohandDark.mp4",
         "VeryLowLight.mp4"]

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE = (8, 8)
GAMMA = 0.5
CONF = 0.3              # matches the truth generation in gt_clips.py


def cond_raw(frame):
    return frame


def cond_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


_GAMMA_LUT = np.clip((np.arange(256) / 255.0) ** GAMMA * 255.0, 0, 255).astype(np.uint8)


def cond_gamma(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], _GAMMA_LUT)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


CONDITIONS = {"raw": cond_raw, "clahe": cond_clahe, "gamma": cond_gamma}


def run(clip, cond, static):
    hands_mod = mp.solutions.hands
    rows = []
    n = 0
    present = 0
    t0 = time.time()
    with hands_mod.Hands(static_image_mode=static, max_num_hands=1,
                         model_complexity=1, min_detection_confidence=CONF,
                         min_tracking_confidence=CONF) as hands:
        cap = cv2.VideoCapture(os.path.join(HERE, clip))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            rgb = cv2.cvtColor(CONDITIONS[cond](frame), cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = np.array([[p.x, p.y] for p in
                               res.multi_hand_landmarks[0].landmark], dtype=np.float32)
                h, w = frame.shape[:2]
                palm = lm[[0, 5, 9, 13, 17]]
                cx, cy = float(palm[:, 0].mean() * w), float(palm[:, 1].mean() * h)
                pw = float(np.hypot((lm[5, 0] - lm[0, 0]) * w,
                                    (lm[5, 1] - lm[0, 1]) * h))
                rows.append((n, 1, cx, cy, pw))
                present += 1
            else:
                rows.append((n, 0, np.nan, np.nan, np.nan))
        cap.release()
    arr = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)
    print(f"  {clip:<22} {cond:<6} {'static' if static else 'track ':<6} "
          f"{n:>4} frames, present {present:>4} ({100*present/max(n,1):>5.1f}%) "
          f"in {time.time()-t0:>5.0f}s", flush=True)
    return arr


def main():
    out = {}
    for cond in CONDITIONS:
        for static in (False, True):
            mode = "static" if static else "track"
            for clip in CLIPS:
                out[f"{cond}_{mode}_{clip}"] = run(clip, cond, static)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"\nsaved {len(out)} arrays -> {OUT}")


if __name__ == "__main__":
    main()
