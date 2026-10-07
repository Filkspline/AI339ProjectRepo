"""Reusable hand-tracking pipeline (PalmDetector + HandLandmarks) with a
MediaPipe-faithful rotated hand crop and correct inverse coordinate mapping.

The coordinate mapping is the important part: MediaPipe rotates the hand to an
upright canonical orientation (from palm keypoints 0=wrist and 2=middle-MCP)
before running the landmark model, and scales/shifts the ROI in the ROTATED
frame.  The repo's PalmDetector._decode_boxes bakes a *2.6 scale and a -h/5.2
shift into the axis-aligned box, so hand_roi() first UNDOES those and then
re-applies MediaPipe's RectTransformation (scale 2.6, shift_y=-0.5) in the
rotated frame.
"""
import os
import sys
from math import pi, floor, cos, sin, atan2

import numpy as np
import cv2 as cv
import torch

from blazepalm import PalmDetector
from handlandmarks import HandLandmarks


def _model_dir():
    """Directory holding the model weight files (bundle-aware).

    In a PyInstaller build the weights are shipped as data files and extracted
    to sys._MEIPASS/models; in a normal source checkout they sit next to this
    module.  Kept separate from the *source* directory so a frozen exe does not
    try to locate weights relative to a dev folder path.
    """
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(__file__)),
                            "models")
    return os.path.dirname(os.path.abspath(__file__))


ML_DIR = _model_dir()
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))

# Standard MediaPipe hand skeleton (21 landmarks, 21 connections).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def normalize_radians(a):
    return a - 2 * pi * floor((a + pi) / (2 * pi))


def letterbox(img, size=256):
    """Resize to fit a size x size box (aspect preserved), pad to square."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    padded = cv.copyMakeBorder(resized, top, bottom, left, right,
                               cv.BORDER_CONSTANT, value=0)
    return padded, scale, left, top


def box_to_pixels(box, scale, left, top):
    """Map a normalized [xmin,ymin,xmax,ymax] box back to original pixels."""
    xmin = (box[0] * 256 - left) / scale
    ymin = (box[1] * 256 - top) / scale
    xmax = (box[2] * 256 - left) / scale
    ymax = (box[3] * 256 - top) / scale
    return xmin, ymin, xmax, ymax


def hand_roi(box_px, kp0_px, kp2_px, scale=2.6, shift_y=-0.5):
    """Return (cx, cy, side, rotation) of the hand ROI in original-image pixels.

    box_px: (xmin, ymin, xmax, ymax) from the repo's decoded box.
    kp0_px, kp2_px: wrist (kp 0) and middle-MCP (kp 2) keypoints in pixels.
    """
    xmin, ymin, xmax, ymax = box_px
    box_w = xmax - xmin
    box_h = ymax - ymin
    # Undo the repo's baked-in scale (*2.6) and axis-aligned shift (-h/5.2):
    w_raw = box_w / 2.6
    h_raw = box_h / 2.6
    cx_raw = (xmin + xmax) / 2.0
    cy_raw = (ymin + ymax) / 2.0 + box_h / 5.2

    x0, y0 = kp0_px
    x1, y1 = kp2_px
    rotation = normalize_radians(pi * 0.5 - atan2(-(y1 - y0), x1 - x0))

    # RectTransformation: scale, shift_y applied in the ROTATED frame.
    shift_x = 0.0
    x_shift = w_raw * shift_x * cos(rotation) - h_raw * shift_y * sin(rotation)
    y_shift = w_raw * shift_x * sin(rotation) + h_raw * shift_y * cos(rotation)
    cx_a = cx_raw + x_shift
    cy_a = cy_raw + y_shift
    long_side = max(w_raw, h_raw)
    side = long_side * scale
    return cx_a, cy_a, side, rotation


def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    p0 = (cx - a * h - b * w, cy + b * h - a * w)
    p1 = (cx + a * h - b * w, cy - b * h - a * w)
    p2 = (2 * cx - p0[0], 2 * cy - p0[1])
    p3 = (2 * cx - p1[0], 2 * cy - p1[1])
    return [p0, p1, p2, p3]  # bottom-left, top-left, top-right, bottom-right


def warp_rect(img, rect_points, out_size=256):
    """Crop the rotated rect and warp it to out_size x out_size.
    Returns (warped, affine_M) where M maps original image -> crop.
    """
    src = np.array([rect_points[1], rect_points[2], rect_points[3]], dtype=np.float32)
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size]], dtype=np.float32)
    M = cv.getAffineTransform(src, dst)
    return cv.warpAffine(img, M, (out_size, out_size)), M


def landmarks_to_image(landmarks_norm, M, out_size=256):
    """Map crop-normalized [0,1] landmarks back to original-image pixels."""
    pts = (landmarks_norm[:, :2] * out_size).astype(np.float32).reshape(-1, 1, 2)
    return cv.transform(pts, cv.invertAffineTransform(M)).reshape(-1, 2)


def draw_hand(frame, hand, draw_box=False):
    """Draw the detection box + landmark skeleton on a BGR frame (in place)."""
    if draw_box:
        xmin, ymin, xmax, ymax = hand["box"]
        cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                     (0, 255, 0), 2)
    pts = hand["landmarks"]  # (21,2) pixels
    for (a, b) in HAND_CONNECTIONS:
        pa = (int(pts[a][0]), int(pts[a][1]))
        pb = (int(pts[b][0]), int(pts[b][1]))
        cv.line(frame, pa, pb, (255, 0, 0), 2)
    for p in pts:
        cv.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    return frame


class HandTracker:
    """Loads both models and runs the full pipeline on BGR frames.

    Includes a near-field false-positive gate: detections whose palm box is
    smaller than `min_box_width` of the frame width are dropped before the
    temporal gate ever sees them.  This suppresses unreliable far-range
    detections (at range/low-light they are indistinguishable from room clutter
    at the detector level); it does NOT solve far/dark hand detection.
    """

    def __init__(self, ml_dir=None, min_score=0.5, min_box_width=0.15):
        ml_dir = ml_dir or ML_DIR
        self.detector = PalmDetector()
        self.detector.load_weights(os.path.join(ml_dir, "palmdetector.pth"))
        self.detector.load_anchors(os.path.join(ml_dir, "anchors.npy"))
        self.detector.eval()

        self.landmarks = HandLandmarks()
        self.landmarks.load_weights(os.path.join(ml_dir, "HandLandmarks.pth"))
        self.landmarks.eval()

        self.min_score = min_score
        self.min_box_width = min_box_width

    def process(self, frame_bgr):
        """Run detection + landmarks on a BGR frame.

        Returns a list of dicts, one per detected hand:
          box        : (xmin, ymin, xmax, ymax) in original-image pixels
          score      : palm detection confidence
          landmarks  : (21, 2) landmark positions in original-image pixels
          presence   : hand-flag score (sigmoid)
          handedness : handedness score (sigmoid, <0.5 ~ left, >0.5 ~ right)
        """
        H, W = frame_bgr.shape[:2]
        padded, scale, left, top = letterbox(frame_bgr, 256)

        with torch.no_grad():
            detections = self.detector.predict_on_image(padded)

        hands = []
        if not detections:
            return hands

        for det in detections[0]:
            score = float(det[18])
            if score < self.min_score:
                continue
            box = box_to_pixels(det[:4].tolist(), scale, left, top)

            # Near-field FP gate: suppress far-range detections.  The decoded
            # box is the palm ROI (2.6x the palm) in original-image pixels;
            # a hand smaller than min_box_width of the frame width is too far
            # / low-light to be distinguishable from clutter, so drop it here
            # (before it could ever reach the consecutive-detection gate).
            xmin, ymin, xmax, ymax = box
            if (xmax - xmin) < self.min_box_width * W:
                continue

            kps = [((float(det[4 + 2 * k]) * 256 - left) / scale,
                    (float(det[5 + 2 * k]) * 256 - top) / scale) for k in range(7)]

            cx, cy, side, rotation = hand_roi(box, kps[0], kps[2])
            rect = rotated_rect_to_points(cx, cy, side, side, rotation)
            crop, M = warp_rect(frame_bgr, rect, 256)

            cropf = torch.from_numpy(crop.astype(np.float32) / 127.5 - 1.0)
            cropf = cropf.permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                hand_flag, handedness, reg = self.landmarks(cropf)

            presence = float(hand_flag[0, 0])
            handedness_v = float(handedness[0, 0])
            lm_norm = reg[0].reshape(21, 3).numpy()      # crop-normalized [0,1]
            lm_px = landmarks_to_image(lm_norm, M, 256)  # (21,2) pixels

            hands.append(dict(
                box=box,
                score=score,
                landmarks=lm_px,
                presence=presence,
                handedness=handedness_v,
                rotation=rotation,
            ))
        return hands
