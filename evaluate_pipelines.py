"""Automated evaluation: our pipeline vs the MediaPipe baseline, on the recorded
clip library.  One command, one CSV, one summary table.

    .venv-blazepalm\\Scripts\\python.exe evaluate_pipelines.py

Both pipelines are measured against the SAME yardstick: MediaPipe's own hands
solution run per-frame in STATIC mode (no tracker state, so no motion lag) is the
ground truth, with the tracking-mode result used only to fill gaps +-5 frames.
That is the arbiter established for the VeryLowLight work, applied to every clip.

  * MediaPipe baseline = MediaPipe hands in TRACKING mode (what a
    `mp.solutions.hands`-style app reports per frame).
  * Our pipeline = hand_mouse_cursor.HandLocator (detector + size gate + motion
    gate + crop verifier), i.e. what the shipped app runs.

Definitions, per clip:
  frames                 decoded frames evaluated
  truth frames           frames where static MediaPipe sees a hand (the yardstick)
  acceptance rate        frames where the pipeline reports ANY position
  true positives         reported position within a hand-scaled radius of a truth
                         hand (radius = max(0.06 frame widths, 0.6 x palm width))
  false positives        reported position with no truth hand near it
  precision              TP / (TP + FP)
  recall                 TP / truth frames
  false-lock rate        on no-hand clips: accepted frames / all frames (any
                         accepted position there is a false lock by construction)

Ground truth comes from cached per-frame MediaPipe runs (gt_clips.npz,
gt_verylowlight.npz) produced by gt_clips.py / gt_verylowlight.py in the mediapipe
venv, so this harness itself only needs the PyTorch environment.

Outputs: build/pipeline_comparison.csv and build/pipeline_summary.txt
"""
import argparse
import csv
import os
import sys
import time

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

CLIPS = [
    ("CloseLightMoving.mp4", "gt_clips.npz", "static:CloseLightMoving.mp4", "hand"),
    ("CloseLightStill.mp4", "gt_clips.npz", "static:CloseLightStill.mp4", "hand"),
    ("CloseDarkStill.mp4", "gt_clips.npz", "static:CloseDarkStill.mp4", "hand"),
    ("FarLight.mp4", "gt_clips.npz", "static:FarLight.mp4", "hand"),
    ("FarDark.mp4", "gt_clips.npz", "static:FarDark.mp4", "hand"),
    ("NohandLight.mp4", "gt_clips.npz", "static:NohandLight.mp4", "nohand"),
    ("NohandDark.mp4", "gt_clips.npz", "static:NohandDark.mp4", "nohand"),
    ("VeryLowLight.mp4", "gt_verylowlight.npz", "static_conf0.3", "hand"),
]
TRACK_KEYS = {"gt_verylowlight.npz": "conf0.3"}
THRESHOLD = [None]
CONSENSUS = [0, hmc.ACQUIRE_CONSENSUS_TOL, 0, hmc.TRACK_CONSENSUS_TOL]
#            ^N  ^acquire tol            ^N  ^track tol  (N = 0 means off)
MATCH_FLOOR = 0.06          # frame widths; raised to 0.6 x palm width when known


def truth_at(gt_static, gt_track, n, window=5):
    """Ground-truth hand position(s) for frame n, from the lag-free run."""
    out = []
    for gt in (gt_static, gt_track):
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


def matched(pos, hands, fw):
    for gx, gy, gpw in hands:
        radius = max(MATCH_FLOOR, 0.6 * gpw / fw)
        if float(np.hypot(pos[0] - gx, pos[1] - gy)) / fw <= radius:
            return True
    return False


def eval_ours(det, clip, gt_static, gt_track, verifier, blob_verifier,
              use_blobs, frames):
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=verifier,
                      motion_gate=MotionGate(), use_motion_blobs=use_blobs,
                      blob_verifier=blob_verifier,
                      acquire_consensus=CONSENSUS[0],
                      acquire_consensus_tol=CONSENSUS[1],
                      track_consensus=CONSENSUS[2],
                      track_consensus_tol=CONSENSUS[3],
                      verifier_threshold=(THRESHOLD[0] if THRESHOLD[0]
                                          else hmc.VERIFIER_THRESHOLD))
    n = tp = fp = accepted = truth = 0
    t0 = time.time()
    dts = []
    prev = None
    while True:
        ret, frame = cap.read()
        if not ret or n >= frames:
            break
        n += 1
        now = time.time()
        dts.append((now - prev) if prev is not None else 0.0)
        prev = now
        fw = frame.shape[1]
        pos, _info = loc.update(frame, now, dts[-1] or 1.0 / 30.0)
        # A frame counts as a truth frame when a hand position is available for
        # it under the SAME borrow rule the matching uses, so recall cannot
        # exceed 100%.
        # STATIC ONLY, and identical for both pipelines: borrowing from the
        # tracking run would score the MediaPipe baseline partly against itself,
        # and any difference in the truth set would make the two columns
        # incomparable.
        hands = truth_at(gt_static, None, n)
        if hands:
            truth += 1
        if pos is not None:
            accepted += 1
            if matched(pos, hands, fw):
                tp += 1
            else:
                fp += 1
    cap.release()
    elapsed = time.time() - t0
    return dict(frames=n, truth_frames=truth, accepted=accepted, tp=tp, fp=fp,
                fps=n / max(elapsed, 1e-6), ms_per_frame=1000 * elapsed / max(n, 1))


def eval_mediapipe(clip, gt_static, gt_track, frames):
    """The baseline: MediaPipe hands in tracking mode."""
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    n = tp = fp = accepted = truth = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or n >= frames:
            break
        n += 1
        fw = frame.shape[1]
        row = gt_track[gt_track[:, 0] == n] if gt_track is not None else []
        pos = None
        if len(row) and row[0, 1]:
            pos = (float(row[0, 2]), float(row[0, 3]))
        hands = truth_at(gt_static, None, n)
        if hands:
            truth += 1
        if pos is not None:
            accepted += 1
            if matched(pos, hands, fw):
                tp += 1
            else:
                fp += 1
    cap.release()
    elapsed = time.time() - t0
    return dict(frames=n, truth_frames=truth, accepted=accepted, tp=tp, fp=fp,
                fps=float("nan"),          # mediapipe decode speed != inference
                ms_per_frame=1000 * elapsed / max(n, 1))


def row_for(clip, kind, label, r):
    prec = r["tp"] / r["accepted"] if r["accepted"] else float("nan")
    rec = r["tp"] / r["truth_frames"] if r["truth_frames"] else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) > 0 else float("nan"))
    return dict(clip=clip, clip_kind=kind, pipeline=label,
                frames=r["frames"],
                truth_frames=r["truth_frames"],
                acceptance_rate_pct=round(100 * r["accepted"] / max(r["frames"], 1), 1),
                true_positives=r["tp"], false_positives=r["fp"],
                precision_pct=(round(100 * prec, 1) if prec == prec else ""),
                recall_pct=(round(100 * rec, 1) if rec == rec else ""),
                f1_pct=(round(100 * f1, 1) if f1 == f1 else ""),
                false_lock_rate_pct=(round(100 * r["accepted"] / max(r["frames"], 1), 1)
                                     if kind == "nohand" else ""),
                ms_per_frame=round(r["ms_per_frame"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--verifier", default="verifier_cnn.pt")
    ap.add_argument("--detector", default=None,
                    help="detector weights (default: palmdetector.pth)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="verifier P(hand) threshold (default: the app's)")
    ap.add_argument("--motion-blobs", action="store_true",
                    help="also evaluate our pipeline with motion blobs enabled")
    ap.add_argument("--variants", action="store_true",
                    help="add extra rows for our pipeline with motion blobs ON "
                         "and with the VeryLowLight-tuned verifier, so the A/B "
                         "options appear in the same table")
    ap.add_argument("--vll-verifier", default="verifier_cnn_vll.pt")
    ap.add_argument("--acquire-consensus", type=int, default=0, metavar="N",
                    help="require N consecutive spatially-agreeing REACQUIRE "
                         "passes before committing to a lock (0 = off)")
    ap.add_argument("--acquire-consensus-tol", type=float,
                    default=hmc.ACQUIRE_CONSENSUS_TOL, metavar="FRAC",
                    help="agreement radius as a fraction of frame width")
    ap.add_argument("--track-consensus", type=int, default=0, metavar="N",
                    help="hold the cursor on an unconfirmed in-window jump until N "
                         "consecutive candidates agree on the new place (0 = off)")
    ap.add_argument("--track-consensus-tol", type=float,
                    default=hmc.TRACK_CONSENSUS_TOL, metavar="FRAC",
                    help="how far a TRACK candidate may jump before it needs "
                         "confirming, as a fraction of frame width")
    ap.add_argument("--out-csv", default=os.path.join(HERE, "build",
                                                      "pipeline_comparison.csv"))
    ap.add_argument("--out-txt", default=os.path.join(HERE, "build",
                                                      "pipeline_summary.txt"))
    args = ap.parse_args()

    THRESHOLD[0] = args.threshold
    CONSENSUS[0] = args.acquire_consensus
    CONSENSUS[1] = args.acquire_consensus_tol
    CONSENSUS[2] = args.track_consensus
    CONSENSUS[3] = args.track_consensus_tol
    det = PalmDetector()
    det_weights = args.detector or os.path.join(ML, "palmdetector.pth")
    if not os.path.isabs(det_weights) and not os.path.isfile(det_weights):
        det_weights = os.path.join(ML, det_weights)
    det.load_weights(det_weights)
    print(f"detector weights      : {det_weights}")
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    verifier = make_verifier(os.path.join(HERE, args.verifier))
    blob_verifier, blob_path = load_blob_verifier(None)
    print(f"our pipeline verifier : {args.verifier} "
          f"@ thr {args.threshold or hmc.VERIFIER_THRESHOLD}")
    print(f"blob verifier         : {blob_path or 'none'}")
    print(f"motion blobs          : "
          f"{'ON' if args.motion_blobs else 'off'}")
    print(f"acquire consensus     : "
          f"{args.acquire_consensus}-pass @ {args.acquire_consensus_tol} fw"
          f"{'' if args.acquire_consensus else ' (off)'}")
    print(f"track consensus       : "
          f"{args.track_consensus}-pass @ {args.track_consensus_tol} fw"
          f"{'' if args.track_consensus else ' (off)'}\n")

    # label our rows with the actual configuration, so the CSV is self-describing
    ours_label = f"Ours ({os.path.basename(args.verifier)}"
    ours_label += (f" @{args.threshold})" if args.threshold
                   else " @app default)")
    if args.acquire_consensus:
        ours_label += f" + consensus {args.acquire_consensus}@{args.acquire_consensus_tol}"
    if args.track_consensus:
        ours_label += f" + trackhold {args.track_consensus}@{args.track_consensus_tol}"
    rows = []
    for clip, gtf, key, kind in CLIPS:
        path = os.path.join(HERE, gtf)
        if not os.path.isfile(path) or not os.path.isfile(os.path.join(HERE, clip)):
            print(f"   {clip:<24} skipped (missing clip or ground truth)")
            continue
        z = np.load(path)
        gt_static = z[key]
        # the baseline is MediaPipe in TRACKING mode: for the original clips that
        # is the clip-named array, for VeryLowLight the conf0.3 tracking run
        if "conf0.3" in z.files and key.startswith("static_conf"):
            gt_track = z["conf0.3"]
        elif clip in z.files:
            gt_track = z[clip]
        else:
            gt_track = gt_static
        rmp = eval_mediapipe(clip, gt_static, gt_track, args.frames)
        rour = eval_ours(det, clip, gt_static, gt_track, verifier, blob_verifier,
                         args.motion_blobs, args.frames)
        rows.append(row_for(clip, kind, "MediaPipe (tracking)", rmp))
        rows.append(row_for(clip, kind, ours_label, rour))
        if args.variants:
            rblob = eval_ours(det, clip, gt_static, gt_track, verifier,
                              blob_verifier, True, args.frames)
            rows.append(row_for(clip, kind,
                                f"Ours + motion blobs ({os.path.basename(args.verifier)})",
                                rblob))
            vll_path = os.path.join(HERE, args.vll_verifier)
            if os.path.isfile(vll_path):
                rvll = eval_ours(det, clip, gt_static, gt_track,
                                 make_verifier(vll_path), blob_verifier, False,
                                 args.frames)
                rows.append(row_for(clip, kind, "Ours + VLL verifier", rvll))
        print(f"   {clip:<24} done "
              f"(truth frames {rour['truth_frames']}, "
              f"ours accepted {rour['accepted']}, mp accepted {rmp['accepted']})",
              flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- presentable summary -------------------------------------------
    lines = []
    add = lines.append
    add("PIPELINE COMPARISON -- recorded clip library")
    add("=" * 100)
    add("Our pipeline: BlazePalm palm detector -> acquisition size+motion gate ->")
    add("              CNN crop verifier -> search-window tracking (the shipped app).")
    add("MediaPipe baseline: mp.solutions.hands in tracking mode, per frame.")
    add("Ground truth for BOTH: MediaPipe static mode (lag-free, per-frame), gaps")
    add("filled from +-5 frames. A reported position counts as correct when it is")
    add("within max(0.06 frame widths, 0.6 x palm width) of a truth hand.")
    add("")
    hdr = (f"{'clip':<22}{'condition':<9}{'pipeline':<22}"
           f"{'accept%':>8}{'precision%':>12}{'recall%':>9}{'F1%':>7}"
           f"{'false locks%':>13}{'ms/frame':>10}")
    add(hdr)
    add("-" * len(hdr))
    for r in rows:
        add(f"{r['clip']:<22}{r['clip_kind']:<9}{r['pipeline']:<22}"
            f"{r['acceptance_rate_pct']:>8.1f}"
            f"{(r['precision_pct'] if r['precision_pct'] != '' else '-'):>12}"
            f"{(r['recall_pct'] if r['recall_pct'] != '' else '-'):>9}"
            f"{(r['f1_pct'] if r['f1_pct'] != '' else '-'):>7}"
            f"{(r['false_lock_rate_pct'] if r['false_lock_rate_pct'] != '' else '-'):>13}"
            f"{r['ms_per_frame']:>10.1f}")

    # aggregate over clips that have truth
    add("")
    add("AGGREGATE (hand clips only; no-hand clips are scored by false-lock rate)")
    add("-" * len(hdr))
    pipes = ["MediaPipe (tracking)", ours_label]
    if args.variants:
        pipes += [f"Ours + motion blobs ({os.path.basename(args.verifier)})",
                  "Ours + VLL verifier"]
    for pipe in pipes:
        sub = [r for r in rows if r["pipeline"] == pipe and r["clip_kind"] == "hand"]
        tp = sum(r["true_positives"] for r in sub)
        fp = sum(r["false_positives"] for r in sub)
        truth = sum(r["truth_frames"] for r in sub)
        frames = sum(r["frames"] for r in sub)
        prec = 100 * tp / (tp + fp) if (tp + fp) else float("nan")
        rec = 100 * tp / truth if truth else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        nohand = [r for r in rows if r["pipeline"] == pipe
                  and r["clip_kind"] == "nohand"]
        fl = (sum(r["acceptance_rate_pct"] * r["frames"] for r in nohand)
              / sum(r["frames"] for r in nohand)) if nohand else float("nan")
        add(f"{pipe:<22} hand clips: truth frames {truth}, accepted "
            f"{tp + fp} ({100*(tp+fp)/max(frames,1):.1f}% of frames) | "
            f"precision {prec:.1f}% recall {rec:.1f}% F1 {f1:.1f}% | "
            f"no-hand false-lock rate {fl:.1f}%")
    add("")
    add("Note: no-hand clips are CloseLightMoving's static counterparts only in the")
    add("sense of containing no hand at all (NohandLight/NohandDark); any accepted")
    add("position there is a false lock by construction.")

    txt = "\n".join(lines)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\nCSV  -> {args.out_csv}\nTEXT -> {args.out_txt}")


if __name__ == "__main__":
    main()
