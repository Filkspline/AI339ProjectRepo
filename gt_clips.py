"""Per-frame MediaPipe ground truth for the recorded clips.

Run with the OLD venv (the one that has mediapipe):
    .venv\\Scripts\\python.exe gt_clips.py

Why this exists: the A/B of the verifier showed FarLight accepted frames falling
45.1% -> 15.9% and FarDark 42.2% -> 0.9%.  That is only a regression if the
frames v2 accepted were really the hand.  The user's live report was that at
range the locator "locks onto real objects in the room" -- i.e. those accepted
frames may themselves have been the false positives.  MediaPipe's own hands
solution gives an independent opinion on where the hand actually is, so we can
label each accepted frame as matched-to-hand or not.

Writes gt_clips.npz: for each clip, an (N, 4) array of
    [frame_index, present, cx_px, cy_px, palm_width_px]
where present is 1 only if a hand was found.
"""
import os

import cv2
import numpy as np
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
         "FarLight.mp4", "FarDark.mp4", "NohandLight.mp4", "NohandDark.mp4"]
MAX_FRAMES = 300


def main():
    mp_hands = mp.solutions.hands
    out = {}
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        model_complexity=1,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3) as hands:
        for clip in CLIPS:
            cap = cv2.VideoCapture(os.path.join(HERE, clip))
            rows = []
            n = 0
            n_present = 0
            while True:
                ret, frame = cap.read()
                if not ret or n >= MAX_FRAMES:
                    break
                n += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm = np.array([[p.x, p.y] for p in
                                   res.multi_hand_landmarks[0].landmark],
                                  dtype=np.float32)
                    h, w = frame.shape[:2]
                    xs, ys = lm[:, 0] * w, lm[:, 1] * h
                    # palm centre = mean of wrist + the four MCP joints
                    palm = lm[[0, 5, 9, 13, 17]]
                    cx, cy = float(palm[:, 0].mean() * w), float(palm[:, 1].mean() * h)
                    # palm width proxy: distance wrist -> index MCP
                    pw = float(np.hypot((lm[5, 0] - lm[0, 0]) * w,
                                        (lm[5, 1] - lm[0, 1]) * h))
                    rows.append((n, 1, cx, cy, pw))
                    n_present += 1
                else:
                    rows.append((n, 0, np.nan, np.nan, np.nan))
            cap.release()
            out[clip] = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5))
            print(f"{clip:<22}{n:>5} frames, hand present {n_present:>4} "
                  f"({100*n_present/max(n,1):.1f}%)")

    np.savez(os.path.join(HERE, "gt_clips.npz"), **out)
    print("\nsaved -> gt_clips.npz")


if __name__ == "__main__":
    main()
