"""Step 1 arbiter: which pipeline is actually on the hand in VeryLowLight?

The offset/coherence numbers were ambiguous -- our palm detector fires
confidently but usually not where MediaPipe's palm centre is, and MediaPipe's own
static detections barely move (median 0.003 frame widths per frame), which is odd
for a clip of a moving hand.  So neither side can be trusted as truth by
assumption.

This asks the landmark model -- a different network, trained to answer exactly
"is this crop a hand?" -- to adjudicate three candidate crops per frame:

    A. our top palm detection, when it is NEAR MediaPipe's palm centre
    B. our top palm detection, when it is FAR from it
    C. a crop centred on MediaPipe's own palm centre (2.6x convention)

`hand` from the landmark model is its presence score (measured earlier: 0.85-1.00
on real close-range hands, ~0.03 on clutter).  Whichever side's crops score high
is the side that is really looking at the hand.

Usage:
    .venv-blazepalm\\Scripts\\python.exe arbiter_vll.py
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
from handlandmarks import HandLandmarks                     # noqa: E402
import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (letterbox, box_to_pixels, hand_roi,  # noqa: E402
                               rotated_rect_to_points, warp_rect)

CLIP = "VeryLowLight.mp4"


def crop_at(frame, cx, cy, side):
    rect = rotated_rect_to_points(cx, cy, side, side, 0.0)
    return warp_rect(frame, rect, 256)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--gt-key", default="static_conf0.3")
    ap.add_argument("--box-scale", type=float, default=2.6,
                    help="ROI side as a multiple of the palm width")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many frames (control runs)")
    args = ap.parse_args()

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    lm_model = HandLandmarks()
    lm_model.load_weights(os.path.join(ML, "HandLandmarks.pth"))
    lm_model.eval()

    z = np.load(os.path.join(HERE, args.gt))
    gt = z[args.gt_key]
    # MediaPipe's own 21 landmarks, so a crop can be anchored on ITS hand using
    # the SAME rotation recipe the pipeline uses.  Without the rotation the crop
    # is not canonical and the landmark model returns ~0 even on a hand (proved
    # by the CloseDarkStill control), which would make this whole test useless.
    lm_key = f"{args.gt_key}_landmarks"
    mp_lm = z[lm_key] if lm_key in z.files else None
    if mp_lm is None:
        print(f"NOTE: {lm_key} not in {args.gt} -- MediaPipe-anchored crops will "
              f"use no rotation and are NOT trustworthy.")

    def presence(crop):
        x = torch.from_numpy(crop[None]).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.no_grad():
            hand, handed, _ = lm_model(x)
        return float(hand[0]), float(handed[0])

    cap = cv2.VideoCapture(os.path.join(HERE, args.clip))
    near_p, far_p, mp_p = [], [], []
    near_h, far_h, mp_h = [], [], []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or (args.limit and n >= args.limit):
            break
        n += 1
        fh, fw = frame.shape[:2]
        padded, scale, left, top = letterbox(frame, 256)
        with torch.no_grad():
            dets = det.predict_on_image(padded)
        dets = [d for d in (dets[0] if dets else []) if float(d[18]) >= hmc.MIN_SCORE]
        row = gt[gt[:, 0] == n]
        mp_pt = (float(row[0, 2]), float(row[0, 3])) if len(row) and row[0, 1] else None
        mp_pw = float(row[0, 4]) if len(row) and row[0, 1] else None

        if dets:
            d = max(dets, key=lambda x: float(x[18]))
            box = box_to_pixels(d[:4].tolist(), scale, left, top)
            kp0 = ((float(d[4]) * 256 - left) / scale, (float(d[5]) * 256 - top) / scale)
            kp2 = ((float(d[8]) * 256 - left) / scale, (float(d[9]) * 256 - top) / scale)
            cx, cy, side, rot = hand_roi(box, kp0, kp2)
            rect = rotated_rect_to_points(cx, cy, side, side, rot)
            crop, _ = warp_rect(frame, rect, 256)
            p, h = presence(crop)
            bc = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            if mp_pt is not None:
                off = float(np.hypot(bc[0] - mp_pt[0], bc[1] - mp_pt[1])) / fw
                (near_p if off <= 0.05 else far_p).append(p)
                (near_h if off <= 0.05 else far_h).append(h)

        if mp_pt is not None:
            # Anchor on MediaPipe's hand with the pipeline's own recipe: wrist
            # (landmark 0) and middle MCP (landmark 9) give the rotation, and a
            # synthetic 2.6x box around the palm centroid feeds hand_roi, which
            # undoes the scale and re-applies the shift exactly as usual.
            if mp_lm is not None and not np.isnan(mp_lm[n - 1]).any():
                L = mp_lm[n - 1] * np.array([fw, fh], dtype=np.float32)
                kp0, kp2 = (float(L[0, 0]), float(L[0, 1])), (float(L[9, 0]), float(L[9, 1]))
                pw = float(np.hypot(kp2[0] - kp0[0], kp2[1] - kp0[1]))
                box = (mp_pt[0] - 1.3 * pw, mp_pt[1] - 1.3 * pw,
                       mp_pt[0] + 1.3 * pw, mp_pt[1] + 1.3 * pw)
                cx, cy, side, rot = hand_roi(box, kp0, kp2)
                rect = rotated_rect_to_points(cx, cy, side, side, rot)
                c, _ = warp_rect(frame, rect, 256)
            else:
                c = crop_at(frame, mp_pt[0], mp_pt[1], args.box_scale * mp_pw)
            p, h = presence(c)
            mp_p.append(p)
            mp_h.append(h)
    cap.release()

    def stat(name, v, h):
        if not v:
            print(f"   {name:<46} (no frames)")
            return
        v = np.array(v)
        h = np.array(h)
        print(f"   {name:<46} n={len(v):>4}  presence median {np.median(v):.3f}  "
              f"p90 {np.percentile(v, 90):.3f}  >0.5: {100*np.mean(v > 0.5):5.1f}%  "
              f"| handedness {np.median(h):.2f}")

    print(f"== landmark-model presence (a DIFFERENT network) as arbiter: "
          f"{args.clip} ==")
    print("   (presence is the landmark model's 'is this crop a hand?' head;\n"
          "    'near' = our box centre within 0.05 frame widths of MediaPipe's)\n")
    stat("A. OUR detection crop, near MediaPipe", near_p, near_h)
    stat("B. OUR detection crop, far from MediaPipe", far_p, far_h)
    stat("C. crop at MEDIAPIPE's palm centre", mp_p, mp_h)

    print("\n   Interpretation: whichever column is high is the side actually\n"
          "   looking at a hand.  A high B means our detector found hands/parts\n"
          "   MediaPipe is not reporting; a low B means our detector is firing on\n"
          "   things that are not hands.")


if __name__ == "__main__":
    main()
