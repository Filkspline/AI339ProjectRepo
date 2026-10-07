"""Step 3, integrated: does the retrained verifier actually help on the clip,
and how does the whole chain compare with MediaPipe on the same frames?

Runs the real v6 pipeline (detector + size gate + motion gate + CNN verifier)
over VeryLowLight.mp4 twice -- once with the previous verifier, once with the
retrained one -- and scores every accepted position against MediaPipe:

    hit          accepted position within a hand-scaled radius of a MediaPipe
                 hand position (borrowing from +-5 frames, because MediaPipe
                 static mode only sees this hand in ~45% of frames)
    clutter lock accepted position with no MediaPipe hand there

Results are also restricted to the HELD-OUT split blocks (every 3rd block of 50
frames, matching build_vll_dataset.py), so the headline number is not measured on
frames the retrained model was trained on.

MediaPipe's own detection rate on the same frames is printed alongside, since
that is the bar to clear.

Usage:
    .venv-blazepalm\\Scripts\\python.exe eval_vll_pipeline.py
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
from hand_mouse_cursor import HandLocator, PlausibilityGate, MotionGate  # noqa: E402
from crop_verifier import make_verifier                     # noqa: E402
from hand_mouse_cursor import load_blob_verifier            # noqa: E402

CLIP = "VeryLowLight.mp4"
DT = 1.0 / 15.0          # this clip is 15 fps
BLOCK, HOLDOUT_EVERY = 50, 3
MATCH_R = 0.06           # frame widths floor; scaled by palm width at match time


def is_holdout(n):
    return (n // BLOCK) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1


def hands_at(gt_s, gt_t, n, window=5):
    """MediaPipe hand positions near frame n from both modes (lag-free first)."""
    out = []
    for gt in (gt_s, gt_t):
        if gt is None:
            continue
        row = gt[gt[:, 0] == n]
        if len(row) and row[0, 1]:
            out.append((float(row[0, 2]), float(row[0, 3]), float(row[0, 4])))
            continue
        present = gt[gt[:, 1] == 1][:, 0]
        if len(present):
            near = present[np.abs(present - n) <= window]
            if len(near):
                k = near[np.argmin(np.abs(near - n))]
                row = gt[gt[:, 0] == k]
                out.append((float(row[0, 2]), float(row[0, 3]), float(row[0, 4])))
    return out


def run(det, verifier, gt_s, gt_t, max_frames, motion_blobs=False,
        blob_verifier=None):
    cap = cv2.VideoCapture(os.path.join(HERE, CLIP))
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=verifier,
                      motion_gate=MotionGate(),
                      use_motion_blobs=motion_blobs,
                      blob_verifier=blob_verifier)
    n = 0
    t = 0.0
    res = []
    prev_mode = loc.mode
    while True:
        ret, frame = cap.read()
        if not ret or n >= max_frames:
            break
        n += 1
        t += DT
        fw = frame.shape[1]
        pos, info = loc.update(frame, t, DT)
        acq = info["mode"] == hmc.TRACK and prev_mode == hmc.REACQUIRE
        prev_mode = info["mode"]
        matched = None
        if pos is not None:
            matched = False
            for gx, gy, gpw in hands_at(gt_s, gt_t, n):
                radius = max(MATCH_R, 0.6 * gpw / fw)
                if float(np.hypot(pos[0] - gx, pos[1] - gy)) / fw <= radius:
                    matched = True
                    break
        res.append(dict(n=n, holdout=is_holdout(n), mode=info["mode"],
                        accepted=pos is not None, matched=bool(matched),
                        acq=acq, blob=info.get("locked_on_blob", False)))
    cap.release()
    return res


def summarise(tag, res, n_mp_static, n_mp_track, note=""):
    def stats(rows):
        n = len(rows)
        acc = sum(1 for r in rows if r["accepted"])
        hit = sum(1 for r in rows if r["matched"])
        fp = acc - hit
        trk = sum(1 for r in rows if r["mode"] == hmc.TRACK)
        acq = sum(1 for r in rows if r["acq"])
        return n, acc, hit, fp, trk, acq

    n, acc, hit, fp, trk, acq = stats(res)
    ho = [r for r in res if r["holdout"]]
    nh, acch, hith, fph, trkh, acqh = stats(ho)
    print(f"\n== {tag} {note}==")
    print(f"   all frames      : {n:>4} frames | accepted {acc:>4} "
          f"({100*acc/max(n,1):5.1f}%) | on the hand {hit:>4} "
          f"({100*hit/max(n,1):5.1f}%) | clutter locks {fp:>4} | "
          f"in TRACK {100*trk/max(n,1):5.1f}% | acquisitions {acq}")
    print(f"   HELD-OUT only   : {nh:>4} frames | accepted {acch:>4} "
          f"({100*acch/max(nh,1):5.1f}%) | on the hand {hith:>4} "
          f"({100*hith/max(nh,1):5.1f}%) | clutter locks {fph:>4} | "
          f"in TRACK {100*trkh/max(nh,1):5.1f}% | acquisitions {acqh}")
    print(f"   MediaPipe, same frames: detects a hand in "
          f"{100*n_mp_static/max(n,1):.1f}% (static) / "
          f"{100*n_mp_track/max(n,1):.1f}% (tracking) of ALL frames")
    if nh:
        hs = sum(1 for r in ho if r["holdout"] and r["matched"])
        print(f"   -> on held-out frames our chain puts a verified position on the "
              f"hand in {hs} of {nh} ({100*hs/nh:.1f}%)")
    return hit, fp, hith, fph, nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--old", default="verifier_cnn.pt")
    ap.add_argument("--new", default="verifier_cnn_vll.pt")
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--motion-blobs", action="store_true",
                    help="enable motion-blob candidates")
    ap.add_argument("--no-blob-verifier", action="store_true")
    args = ap.parse_args()

    z = np.load(os.path.join(HERE, args.gt))
    gt_s = z["static_conf0.3"]
    gt_t = z["conf0.3"]

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()

    # MediaPipe's own rate on the same frames (from the GT file itself)
    n_static = int(gt_s[:, 1].sum())
    n_track = int(gt_t[:, 1].sum())
    N = len(gt_s)

    old = make_verifier(os.path.join(HERE, args.old))
    new = make_verifier(os.path.join(HERE, args.new))
    bv, bv_path = (load_blob_verifier(None) if args.motion_blobs
                   and not args.no_blob_verifier else (None, None))
    if args.motion_blobs:
        print(f"blob verifier: {bv_path if bv else 'NONE (main verifier, 8-way)'}")
    r_old = run(det, old, gt_s, gt_t, args.frames, args.motion_blobs, bv)
    r_new = run(det, new, gt_s, gt_t, args.frames, args.motion_blobs, bv)
    if args.motion_blobs:
        nb = sum(1 for r in r_new if r["blob"] and r["accepted"])
        print(f"motion-blob path: {nb} accepted positions came from a blob")

    print(f"MediaPipe on {N} frames of {CLIP}: static {n_static} "
          f"({100*n_static/N:.1f}%), tracking {n_track} ({100*n_track/N:.1f}%)")
    print(f"held-out split: every {HOLDOUT_EVERY}rd block of {BLOCK} frames")
    summarise("PREVIOUS verifier", r_old, n_static, n_track)
    summarise("RETRAINED verifier", r_new, n_static, n_track)
    print("\nNote: 'on the hand' is measured against MediaPipe's own detections, so"
          "\nMediaPipe is the reference, not an independent truth -- our recall can"
          "\nnever exceed it.  The meaningful comparison is that MediaPipe's own"
          "\nrate is the bar, and any position we report off the hand is a lock on"
          "\nsomething else.")


if __name__ == "__main__":
    main()
