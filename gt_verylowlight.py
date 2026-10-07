"""Per-frame MediaPipe ground truth for VeryLowLight.mp4.

Run with the OLD venv (the one that has mediapipe):
    .venv\\Scripts\\python.exe gt_verylowlight.py

Same contract as gt_clips.py: for each frame,
    [frame_index, present, cx_px, cy_px, palm_width_px]
so the diagnosis and the step-3 comparison can use identical matching.

Also sweeps MediaPipe's own min_detection_confidence, because "MediaPipe fails
here" is only useful if we know whether it is a threshold artefact or a genuine
failure at any sane threshold.

Writes gt_verylowlight.npz with one array per clip name.
"""
import os

import cv2
import numpy as np
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = "VeryLowLight.mp4"
CONFS = (0.1, 0.3, 0.5)


def run(clip, conf, max_frames=100000, static=False, want_landmarks=False):
    mp_hands = mp.solutions.hands
    rows = []
    lms = []
    n = 0
    present = 0
    with mp_hands.Hands(static_image_mode=static, max_num_hands=1,
                        model_complexity=1,
                        min_detection_confidence=conf,
                        min_tracking_confidence=conf) as hands:
        cap = cv2.VideoCapture(os.path.join(HERE, clip))
        while True:
            ret, frame = cap.read()
            if not ret or n >= max_frames:
                break
            n += 1
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = np.array([[p.x, p.y] for p in
                               res.multi_hand_landmarks[0].landmark], dtype=np.float32)
                h, w = frame.shape[:2]
                palm = lm[[0, 5, 9, 13, 17]]
                cx, cy = float(palm[:, 0].mean() * w), float(palm[:, 1].mean() * h)
                pw = float(np.hypot((lm[5, 0] - lm[0, 0]) * w,
                                    (lm[5, 1] - lm[0, 1]) * h))
                rows.append((n, 1, cx, cy, pw))
                if want_landmarks:
                    lms.append(lm)
                present += 1
            else:
                rows.append((n, 0, np.nan, np.nan, np.nan))
                if want_landmarks:
                    lms.append(np.full((21, 2), np.nan, dtype=np.float32))
        cap.release()
    print(f"{clip} @ conf {conf}{' static' if static else ''}: {n} frames, "
          f"MP present {present} ({100*present/max(n,1):.1f}%)")
    out = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5))
    if want_landmarks:
        return out, (np.stack(lms) if lms else np.zeros((0, 21, 2), np.float32))
    return out


def main():
    # Tracking mode lags behind a fast-moving hand, which would make any
    # "is our detection on the hand?" test unfair; static mode has no temporal
    # state at all, so it gives a lag-free per-frame reference.  Both are saved,
    # and the static conf-0.3 run also saves all 21 landmarks so a crop can be
    # anchored on MediaPipe's own hand with the correct rotation recipe.
    out = {}
    for conf in CONFS:
        out[f"conf{conf}"] = run(CLIP, conf)
    for conf in CONFS:
        if conf == 0.3:
            arr, lms = run(CLIP, conf, static=True, want_landmarks=True)
            out[f"static_conf{conf}"] = arr
            out[f"static_conf{conf}_landmarks"] = lms
        else:
            out[f"static_conf{conf}"] = run(CLIP, conf, static=True)
    path = os.path.join(HERE, "gt_verylowlight.npz")
    np.savez(path, **out)
    print(f"\nsaved -> {path} (keys: {', '.join(out)})")


if __name__ == "__main__":
    main()
