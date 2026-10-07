"""Add lag-free STATIC-mode MediaPipe ground truth for the original 7 clips.

evaluate_pipelines.py needs the same yardstick for every clip. gt_clips.npz only
holds tracking-mode results, and tracking mode IS the MediaPipe baseline -- scoring
it against itself reports a meaningless 100% precision/recall. This run adds
per-frame static-mode (no tracker state) results under "static:<clip>" keys,
leaving the existing arrays untouched for backwards compatibility.

Run with the OLD venv (the one that has mediapipe):
    .venv\\Scripts\\python.exe gt_clips_static.py
"""
import os

import cv2
import numpy as np
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
         "FarLight.mp4", "FarDark.mp4", "NohandLight.mp4", "NohandDark.mp4"]
MAX_FRAMES = 700
CONF = 0.3


def main():
    z = dict(np.load(os.path.join(HERE, "gt_clips.npz")))
    mp_hands = mp.solutions.hands
    for clip in CLIPS:
        rows = []
        n = 0
        present = 0
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                            model_complexity=1, min_detection_confidence=CONF,
                            min_tracking_confidence=CONF) as hands:
            cap = cv2.VideoCapture(os.path.join(HERE, clip))
            while True:
                ret, frame = cap.read()
                if not ret or n >= MAX_FRAMES:
                    break
                n += 1
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    lm = np.array([[p.x, p.y] for p in
                                   res.multi_hand_landmarks[0].landmark],
                                  dtype=np.float32)
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
        z[f"static:{clip}"] = np.array(rows, dtype=np.float32)
        print(f"{clip:<22} {n:>4} frames, static MP present {present:>4} "
              f"({100*present/max(n,1):.1f}%)")
    np.savez(os.path.join(HERE, "gt_clips.npz"), **z)
    print("\nupdated -> gt_clips.npz (added static:<clip> keys)")


if __name__ == "__main__":
    main()
