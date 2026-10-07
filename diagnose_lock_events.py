"""Count cursor JUMP events, and find out which mode produces them.

F1 on CloseDarkStill barely moved when the consensus was enabled (12.1 -> 16.0),
which is suspicious: the user's complaint is not "fewer correct frames", it is
"the cursor jumps to an armpit and comes back".  Those are different quantities,
and the harness does not measure the second one.

So this measures, per clip and per configuration:

  * accepted frames split by the mode that produced them (REACQUIRE vs TRACK).
    This matters because the consensus was applied to REACQUIRE only, as asked --
    if most false positions come from TRACK, the consensus cannot help much, and
    that is the honest explanation for a flat F1 rather than a mystery.
  * EXCURSIONS: a run of accepted-but-off-the-hand positions (brief 1-2 frame
    gaps ignored) that is preceded by a correctly-placed accepted frame.  That is
    the "locked onto an armpit, then self-corrected" event the user described.
    Reported with the median duration and how many last >=10 frames (persistent,
    i.e. a real wrong lock rather than a flicker).

Run: .venv-blazepalm\\Scripts\\python.exe diagnose_lock_events.py
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

from blazepalm import PalmDetector                                   # noqa: E402
from crop_verifier import make_verifier                              # noqa: E402
from hand_mouse_cursor import (HandLocator, PlausibilityGate,        # noqa: E402
                               MotionGate, load_blob_verifier)
from evaluate_pipelines import CLIPS, matched, truth_at              # noqa: E402

GAP = 2          # ignore gaps of up to this many non-accepted frames
LOOKBACK = 15    # an excursion only counts if the hand was on target recently


def load(det_name, ver_name, thr):
    det = PalmDetector()
    det.load_weights(os.path.join(HERE, det_name))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    return det, make_verifier(os.path.join(HERE, ver_name)), thr


def run_config(det, ver, thr, blobs, consensus, tol, clip, gt_path, key, frames,
               track=0, track_tol=0.06):
    z = np.load(os.path.join(HERE, gt_path))
    gt = z[key]
    blob_ver, _p = load_blob_verifier(None) if blobs else (None, None)
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=ver,
                      verifier_threshold=thr, motion_gate=MotionGate(),
                      use_motion_blobs=blobs, blob_verifier=blob_ver,
                      acquire_consensus=consensus, acquire_consensus_tol=tol,
                      track_consensus=track, track_consensus_tol=track_tol)
    rec = []       # (mode_before, accepted, on_hand)
    n = 0
    prev = None
    import time
    while True:
        ok, fr = cap.read()
        if not ok or n >= frames:
            break
        n += 1
        now = time.time()
        dt = (now - prev) if prev is not None else 1.0 / 30.0
        prev = now
        mode_before = loc.mode
        pos, _info = loc.update(fr, now, dt or 1.0 / 30.0)
        hands = truth_at(gt, None, n)
        on = None if pos is None else (bool(hands) and matched(pos, hands, fr.shape[1]))
        rec.append((mode_before, pos is not None, on,
                    None if pos is None else (float(pos[0]), float(pos[1])),
                    fr.shape[1]))
    cap.release()
    return rec


def track_displacement(rec, label):
    """How far does an accepted position move between consecutive frames, in TRACK?

    This is the measurement the TRACK-side hold needs: the tolerance must be above
    the genuine tracking motion of a hand that is being followed, and below the
    distance at which a candidate is jumping off the hand.  A TRACK tolerance
    cannot be copied from the REACQUIRE one -- a tracked hand is moving all the
    time, so its frame-to-frame steps are larger than an acquisition candidate's.
    """
    tp_tp, tp_fp, fp_tp, fp_fp = [], [], [], []
    prev = None
    for r in rec:
        mode, acc, on = r[0], r[1], r[2]
        pos, fw = r[3], r[4]
        if not acc or mode != "TRACK":
            prev = None if not acc else prev
            continue
        if prev is not None:
            d = float(np.hypot(pos[0] - prev[0][0], pos[1] - prev[0][1])) / fw
            a, b = prev[1], on
            if a and b:
                tp_tp.append(d)
            elif a and not b:
                tp_fp.append(d)
            elif not a and b:
                fp_tp.append(d)
            else:
                fp_fp.append(d)
        prev = (pos, on)

    def q(v, p):
        return float(np.percentile(v, p)) if v else float("nan")

    print(f"    {label}")
    for nm, v in (("on-hand -> on-hand (real tracking motion)", tp_tp),
                  ("on-hand -> off-hand (a JUMP off the hand)", tp_fp),
                  ("off-hand -> on-hand (jump back)", fp_tp),
                  ("off-hand -> off-hand (wandering while wrong)", fp_fp)):
        if not v:
            print(f"      {nm:<44} none")
            continue
        print(f"      {nm:<44} n={len(v):<4} med {q(v, 50):.4f} p75 {q(v, 75):.4f} "
              f"p90 {q(v, 90):.4f} p95 {q(v, 95):.4f} max {max(v):.4f}")
    if tp_tp and tp_fp:
        print(f"      -> jump median {q(tp_fp, 50):.4f} vs real-motion p95 "
              f"{q(tp_tp, 95):.4f}: "
              f"{'SEPARABLE' if q(tp_fp, 50) > q(tp_tp, 95) else 'OVERLAPPING'}"
              f" (jump p10 {q(tp_fp, 10):.4f}, p25 {q(tp_fp, 25):.4f})")


def summarise(rec, label):
    acc = [r for r in rec if r[1]]
    f_re = sum(1 for r in acc if r[0] == "REACQUIRE" and not r[2])
    t_re = sum(1 for r in acc if r[0] == "REACQUIRE" and r[2])
    f_tr = sum(1 for r in acc if r[0] == "TRACK" and not r[2])
    t_tr = sum(1 for r in acc if r[0] == "TRACK" and r[2])
    false_total = f_re + f_tr
    # excursions: maximal runs of non-matched accepted frames with short gaps
    exc = []
    i = 0
    last_good = -10 ** 9
    while i < len(rec):
        a, on = rec[i][1], rec[i][2]
        if a and on:
            last_good = i
            i += 1
            continue
        if a and on is False:
            j = i
            gaps = 0
            end = i
            while j < len(rec):
                a2, on2 = rec[j][1], rec[j][2]
                if a2 and on2 is False:
                    end = j
                    j += 1
                elif not a2 and gaps < GAP:
                    gaps += 1
                    j += 1
                else:
                    break
            if i - last_good <= LOOKBACK:
                exc.append((i, end - i + 1, rec[i][0]))
            i = end + 1
            continue
        i += 1
    durs = [e[1] for e in exc]
    persistent = sum(1 for d in durs if d >= 10)
    print(f"    {label:<34} accepted {len(acc):<4} false {false_total:<4} "
          f"(REACQUIRE {f_re}, TRACK {f_tr})   off-hand {t_re}/{f_re} & {t_tr}/{f_tr}"
          f"   EXCURSIONS {len(exc):<3} median "
          f"{np.median(durs) if durs else 0:.0f}f, >=10f: {persistent}")
    return dict(accepted=len(acc), false=false_total, f_re=f_re, f_tr=f_tr,
                exc=len(exc), persistent=persistent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100000)
    args = ap.parse_args()

    det, ver, thr = load("palmdetector_dark.pth", "verifier_cnn_big.pt", 0.61)
    print("dark config; comparing consensus OFF vs ON (3 @ 0.05 fw)")
    print("'accepted T/F by mode' = on-hand/off-hand accepted frames per mode\n")
    tot = {}
    for clip, gtf, key, kind in CLIPS:
        if not (os.path.isfile(os.path.join(HERE, clip))
                and os.path.isfile(os.path.join(HERE, gtf))):
            continue
        if kind != "hand":
            continue
        print(f"  {clip}")
        for label, cons, track in (("v3: acquire consensus only", 3, 0),
                                   ("v4: + TRACK hold 3 @ 0.06", 3, 3)):
            rec = run_config(det, ver, thr, True, cons, 0.05, clip, gtf, key,
                             args.frames, track=track)
            if cons and not track:
                # the TRACK-side numbers are measured on the config we are about to
                # change, i.e. the consensus build
                track_displacement(rec, f"TRACK displacements ({label})")
            s = summarise(rec, label)
            a = tot.setdefault(label, dict(accepted=0, false=0, exc=0,
                                           persistent=0, f_re=0, f_tr=0))
            for k in a:
                a[k] += s[k]
    print("\n  TOTALS across hand clips")
    for label, a in tot.items():
        print(f"    {label:<34} accepted {a['accepted']:<5} false {a['false']:<4} "
              f"(REACQUIRE {a['f_re']}, TRACK {a['f_tr']})  "
              f"excursions {a['exc']} of which persistent {a['persistent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
