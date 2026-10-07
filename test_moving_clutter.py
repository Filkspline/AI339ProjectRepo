"""Step 3.4: does the motion-blob path lock onto something moving that is NOT a
hand?

None of our recorded clips contain moving clutter (the two no-hand clips are
static rooms), so this failure mode cannot be tested on existing footage -- it has
to be constructed.  This takes NohandLight.mp4 (a real static room, no hand
anywhere) and injects a moving object: a textured patch cut from the frame itself,
translated across the scene.  That is a controlled stand-in for a curtain, a fan
or a passing person -- NOT the real thing, and it is labelled as such.

There is no hand in this clip, so EVERY accepted position is a false lock, and the
test reports whether those locks land on the injected moving object (i.e. the
pipeline followed the motion) or elsewhere.

Usage:
    .venv-blazepalm\\Scripts\\python.exe test_moving_clutter.py
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (HandLocator, PlausibilityGate,  # noqa: E402
                               MotionGate, load_blob_verifier)
from crop_verifier import make_verifier                     # noqa: E402

CLIP = "NohandLight.mp4"
DT = 1.0 / 15.0


def inject(frame, patch, x, y, w, h):
    out = frame.copy()
    out[y:y + h, x:x + w] = patch
    return out


def run(det, verifier, motion_blobs, max_frames, speed, blob_verifier=None):
    cap = cv2.VideoCapture(os.path.join(HERE, CLIP))
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=verifier,
                      motion_gate=MotionGate(),
                      use_motion_blobs=motion_blobs,
                      blob_verifier=blob_verifier)
    n = 0
    accepted = on_object = 0
    acquisitions = blob_acq = 0
    positions = []
    patch = None
    w = h = 0
    t = 0.0
    prev_mode = loc.mode
    while True:
        ret, frame = cap.read()
        if not ret or n >= max_frames:
            break
        n += 1
        t += DT
        fh, fw = frame.shape[:2]
        if patch is None:                      # cut a real texture from frame 1
            w = int(0.15 * fw)
            h = int(0.15 * fh)
            py, px = int(0.25 * fh), int(0.10 * fw)
            patch = frame[py:py + h, px:px + w].copy()
        # keep the patch fully inside the frame and wrap it back to the left
        x = int((speed * n) % max(fw - w, 1))
        y = int(0.25 * fh)
        frame = inject(frame, patch, x, y, w, h)
        pos, info = loc.update(frame, t, DT)
        if info["mode"] == hmc.TRACK and prev_mode == hmc.REACQUIRE:
            acquisitions += 1
            if info.get("locked_on_blob"):
                blob_acq += 1
        prev_mode = info["mode"]
        if pos is not None:
            accepted += 1
            positions.append((pos[0], pos[1]))
            cx, cy = x + w / 2.0, y + h / 2.0
            if float(np.hypot(pos[0] - cx, pos[1] - cy)) / fw <= 0.12:
                on_object += 1
    cap.release()
    return dict(n=n, accepted=accepted, on_object=on_object,
                acquisitions=acquisitions, blob_acq=blob_acq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--verifier", default="verifier_cnn.pt")
    ap.add_argument("--no-blob-verifier", dest="blob_verifier",
                    action="store_false", default=True)
    args = ap.parse_args()

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    verifier = make_verifier(os.path.join(HERE, args.verifier))

    print(f"{CLIP} + an injected moving textured patch "
          f"({args.speed:.0f} px/frame), which contains NO hand.")
    print("Every accepted position below is therefore a FALSE lock.\n")
    print(f"   {'config':<28}{'frames':>7}{'acq':>5}{'from blob':>10}"
          f"{'accepted':>10}{'on the object':>15}")
    bv = load_blob_verifier(None)[0] if args.blob_verifier else None
    for blobs in (False, True):
        r = run(det, verifier, blobs, args.frames, args.speed, bv)
        tag = ("motion blobs ON" + (" + blob verifier" if (blobs and bv)
               else " (main verifier, 8-way)" if blobs
               else "")) if blobs else "motion blobs OFF (v6/v7)"
        print(f"   {tag:<28}{r['n']:>7}{r['acquisitions']:>5}"
              f"{r['blob_acq']:>10}{r['accepted']:>10}"
              f"{r['on_object']:>9} ({100*r['on_object']/max(r['accepted'],1):>3.0f}%)")
    print("\n   'acq' is the number of times the locator COMMITTED to a lock;"
          "\n   one acquisition on the moving object is enough to produce many"
          "\n   accepted frames, because TRACK then holds and follows it.")
    print("\nCaveat: the moving object is a synthetic translating patch cut from the"
          "\nscene, not a real curtain/fan/person. No recorded clip contains real"
          "\nmoving clutter, so that remains untested on real footage.")


if __name__ == "__main__":
    main()
