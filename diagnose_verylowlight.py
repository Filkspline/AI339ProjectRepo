"""Step 1 diagnosis: WHERE does the pipeline fail on VeryLowLight.mp4?

Runs every frame of the target clip through the v6 chain stage by stage and
logs, per frame, whether each stage accepted:

    palm detector (raw, score-only)  ->  size gate (0.10)  ->  motion gate
        ->  CNN verifier P(hand)

plus, independently, whether MediaPipe's own hands solution found a hand in the
same frame (from gt_verylowlight.npz, produced by gt_verylowlight.py in the
mediapipe venv).  The point is to attribute the failure to a stage instead of
guessing that "more verifier training data" is the fix.

The decisive question is not "did our detector fire" but "did it fire ON THE
HAND", so for frames where MediaPipe sees a hand it also records whether our
best detection is anywhere near MediaPipe's palm centre.

Also measures the two things the motion gate design flagged for dark rooms:
sensor noise (median frame-to-frame difference, which is robust to the moving
hand) and the colour cast (per-channel means, compared with the daylight clips).

Outputs build/verylowlight_perframe.csv and a stage summary on stdout.

Usage:
    .venv-blazepalm\\Scripts\\python.exe diagnose_verylowlight.py
"""
import argparse
import csv
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
import hand_mouse_cursor as hmc                             # noqa: E402
from hand_mouse_cursor import (MotionGate, letterbox, box_to_pixels,  # noqa: E402
                               hand_roi, rotated_rect_to_points, warp_rect,
                               blob_candidates, load_blob_verifier)
from crop_verifier import make_verifier                     # noqa: E402

CLIP = "VeryLowLight.mp4"
SCORE_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)
DAYLIGHT = ["NohandLight.mp4", "CloseLightStill.mp4", "NohandDark.mp4",
            "CloseDarkStill.mp4"]


def detection_crop(frame, d, scale, left, top):
    """Rotation-normalised 256x256 crop for one detection (same recipe as the app)."""
    box = box_to_pixels(d[:4].tolist(), scale, left, top)
    kp0 = ((float(d[4]) * 256 - left) / scale, (float(d[5]) * 256 - top) / scale)
    kp2 = ((float(d[8]) * 256 - left) / scale, (float(d[9]) * 256 - top) / scale)
    cx, cy, side, rot = hand_roi(box, kp0, kp2)
    rect = rotated_rect_to_points(cx, cy, side, side, rot)
    crop, _ = warp_rect(frame, rect, 256)
    return box, crop


def boxes_contain(box, pt, pad=0.15):
    """Does `box` (xmin, ymin, xmax, ymax) contain `pt`, with a little slack?"""
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - pad * w <= pt[0] <= box[2] + pad * w
            and box[1] - pad * h <= pt[1] <= box[3] + pad * h)


def clip_colour(path, max_frames=120):
    cap = cv2.VideoCapture(path)
    bgr, lum, diffs, n = [], [], [], 0
    prev = None
    while True:
        ok, f = cap.read()
        if not ok or n >= max_frames:
            break
        n += 1
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bgr.append(f.reshape(-1, 3).mean(axis=0).astype(np.float32))
        lum.append(float(g.mean()))
        if prev is not None:
            diffs.append(float(np.median(np.abs(g - prev))))
        prev = g
    cap.release()
    b, g_, r = np.mean(bgr, axis=0) if bgr else (np.nan,) * 3
    # median |frame - frame| over the whole frame is dominated by the static
    # background (robust to the moving hand); for white noise its expectation is
    # 0.954 sigma, so sigma ~= median / 0.954
    sigma = (float(np.median(diffs)) / 0.954) if diffs else float("nan")
    return dict(n=n, luma=float(np.mean(lum)) if lum else float("nan"),
                b=float(b), g=float(g_), r=float(r), sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--verifier", default=hmc.VERIFIER_FILENAME)
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--gt-key", default="conf0.3",
                    help="which MediaPipe confidence sweep to use as truth")
    ap.add_argument("--match-r", type=float, default=0.12,
                    help="match radius (frame widths) for 'fired on the hand'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=hmc.VERIFIER_THRESHOLD,
                    help="verifier P(hand) threshold for this run")
    ap.add_argument("--motion-blobs", action="store_true",
                    help="also evaluate motion-blob proposals + blob verifier, to "
                         "see whether they recover frames the detector misses")
    ap.add_argument("--blob-threshold", type=float,
                    default=hmc.MOTION_BLOB_THRESHOLD)
    args = ap.parse_args()
    THR = args.threshold

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    verifier = make_verifier(os.path.join(HERE, args.verifier))
    blob_verifier = None
    if args.motion_blobs:
        blob_verifier, bp = load_blob_verifier(None)
        print(f"blob verifier: {bp} @ {args.blob_threshold}")
    print(f"verifier {args.verifier} @ threshold {THR}")

    gt_path = os.path.join(HERE, args.gt)
    gt = None
    if os.path.isfile(gt_path):
        z = np.load(gt_path)
        key = args.gt_key if args.gt_key in z.files else z.files[0]
        gt = z[key]
        print(f"MediaPipe truth: {args.gt}[{key}]")
    if gt is None:
        print(f"NOTE: {args.gt} missing -- MediaPipe columns will be blank.")

    cap = cv2.VideoCapture(os.path.join(HERE, args.clip))
    gate = MotionGate()             # fed in order, exactly as the live pipeline does
    rows = []
    prev_luma = None
    prev_gray = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or (args.limit and n >= args.limit):
            break
        n += 1
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lm = float(gray.mean())
        gshift = float(abs(lm - prev_luma)) if prev_luma is not None else 0.0
        frame_noise = (float(np.median(np.abs(gray - prev_gray)))
                       if prev_gray is not None else float("nan"))
        prev_luma, prev_gray = lm, gray
        gate.push(frame)

        padded, scale, left, top = letterbox(frame, 256)
        with torch.no_grad():
            dets = det.predict_on_image(padded)
        dets = dets[0] if dets else []

        scores = [float(d[18]) for d in dets]
        counts = {t: sum(1 for s in scores if s >= t) for t in SCORE_SWEEP}
        best_score = max(scores) if scores else 0.0

        # ---- stage funnel ------------------------------------------------
        # Two candidates matter, and they are not the same one:
        #   * the HIGHEST-SCORING detection (what the diagnosis followed first)
        #   * the detection that actually CONTAINS the hand MediaPipe sees
        #     (what we care about -- "did our chain accept the real hand?")
        # The first can be a false positive while the second is genuine, so both
        # are tracked.  Containment, not centre distance: at 15 fps with a hand
        # moving fast, MediaPipe's tracked centre lags, so centre distance alone
        # unfairly marks correct detections as "not on the hand".
        keep = [d for d in dets if float(d[18]) >= hmc.MIN_SCORE]

        # MediaPipe's opinion for this frame (needed before judging candidates)
        mp_present = False
        mp_pt = None
        mp_pw = float("nan")
        if gt is not None:
            row = gt[gt[:, 0] == n]
            if len(row) and row[0, 1]:
                mp_present = True
                mp_pt = (float(row[0, 2]), float(row[0, 3]))
                mp_pw = float(row[0, 4])

        # ---- motion-blob candidates (optional) ---------------------------
        # Does motion PROPOSE the hand where the detector failed, and does the
        # blob verifier CONFIRM it?  Split into proposed-on-hand and
        # confirmed-on-hand so a confirmation failure is distinguishable from a
        # proposal failure.
        blob_proposed = blob_confirmed = False
        n_blob_raw = n_blob_ok = 0
        if blob_verifier is not None:
            raw = gate.blobs(frame.shape)
            n_blob_raw = len(raw)
            if mp_pt is not None:
                for cx, cy, _side, _af in raw:
                    if float(np.hypot(cx - mp_pt[0], cy - mp_pt[1])) / fw <= 0.12:
                        blob_proposed = True
                        break
            conf = blob_candidates(frame, gate, blob_verifier,
                                   args.blob_threshold)
            n_blob_ok = len(conf)
            if mp_pt is not None:
                for box, _p, _info in conf:
                    c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                    if float(np.hypot(c[0] - mp_pt[0], c[1] - mp_pt[1])) / fw <= 0.12:
                        blob_confirmed = True
                        break

        box_w_frac = float("nan")
        box_c = None
        off_norm = float("nan")
        size_ok = motion_ok = verifier_ok = False
        p_hand = m_frac = m_thr = float("nan")
        top_on_hand = False
        hand_det_score = float("nan")
        hand_size_ok = hand_motion_ok = hand_verifier_ok = False
        hand_m_frac = hand_p_hand = float("nan")
        if keep:
            best_d = max(keep, key=lambda d: float(d[18]))
            box, crop = detection_crop(frame, best_d, scale, left, top)
            box_w_frac = (box[2] - box[0]) / fw
            size_ok = box_w_frac >= hmc.MIN_BOX_FRAC_REACQUIRE
            if size_ok:
                motion_ok, minfo = gate.score(box, frame.shape)
                m_frac = minfo["frac"] if minfo["frac"] is not None else float("nan")
                m_thr = minfo["threshold"]
                p_hand = float(verifier(crop))
                verifier_ok = p_hand >= THR
            if mp_pt is not None:
                top_on_hand = boxes_contain(box, mp_pt)
                box_c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                off_norm = (float(np.hypot(box_c[0] - mp_pt[0], box_c[1] - mp_pt[1]))
                            / max(box[2] - box[0], 1e-6))

            if mp_pt is not None:
                # the best-scoring detection whose box holds the hand
                on = [d for d in keep
                      if boxes_contain(box_to_pixels(d[:4].tolist(), scale, left, top),
                                       mp_pt)]
                if on:
                    hd = max(on, key=lambda d: float(d[18]))
                    hand_det_score = float(hd[18])
                    hbox, hcrop = detection_crop(frame, hd, scale, left, top)
                    hand_size_ok = ((hbox[2] - hbox[0]) / fw
                                    >= hmc.MIN_BOX_FRAC_REACQUIRE)
                    if hand_size_ok:
                        hand_motion_ok, hinfo = gate.score(hbox, frame.shape)
                        hand_m_frac = (hinfo["frac"] if hinfo["frac"] is not None
                                       else float("nan"))
                        hand_p_hand = float(verifier(hcrop))
                        hand_verifier_ok = hand_p_hand >= THR

        rows.append(dict(
            frame=n, luma=round(lm, 2), gshift=round(gshift, 2),
            frame_noise=round(frame_noise, 3) if frame_noise == frame_noise else "",
            n_dets=len(dets), best_score=round(best_score, 4),
            n030=counts[0.3], n040=counts[0.4], n050=counts[0.5],
            n060=counts[0.6], n070=counts[0.7],
            box_w_frac=round(box_w_frac, 4) if box_w_frac == box_w_frac else "",
            size_ok=int(size_ok), motion_ok=int(motion_ok),
            m_frac=round(m_frac, 4) if m_frac == m_frac else "",
            m_thr=round(m_thr, 4) if m_thr == m_thr else "",
            p_hand=round(p_hand, 4) if p_hand == p_hand else "",
            verifier_ok=int(verifier_ok), mp=int(mp_present),
            top_on_hand=int(top_on_hand),
            box_c=(round(box_c[0] / fw, 4), round(box_c[1] / fh, 4))
            if box_c is not None else "",
            off_norm=round(off_norm, 4) if off_norm == off_norm else "",
            mp_c=(round(mp_pt[0] / fw, 4), round(mp_pt[1] / fh, 4))
            if mp_pt is not None else "",
            hand_det=1 if hand_det_score == hand_det_score else 0,
            hand_det_score=(round(hand_det_score, 4)
                            if hand_det_score == hand_det_score else ""),
            hand_size_ok=int(hand_size_ok),
            hand_motion_ok=int(hand_motion_ok),
            hand_m_frac=round(hand_m_frac, 4) if hand_m_frac == hand_m_frac else "",
            hand_p_hand=round(hand_p_hand, 4) if hand_p_hand == hand_p_hand else "",
            hand_verifier_ok=int(hand_verifier_ok),
            blob_proposed=int(blob_proposed),
            blob_confirmed=int(blob_confirmed),
            n_blob_raw=n_blob_raw, n_blob_ok=n_blob_ok))
    cap.release()

    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    out = os.path.join(HERE, "build", "verylowlight_perframe.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- summary -------------------------------------------
    N = len(rows)
    pct = lambda k: 100.0 * sum(1 for r in rows if r[k]) / max(N, 1)   # noqa: E731
    best = np.array([r["best_score"] for r in rows])
    lumas = np.array([r["luma"] for r in rows])
    print(f"== {args.clip}: {N} frames, luma {lumas.mean():.1f} "
          f"(min {lumas.min():.1f}, max {lumas.max():.1f}) ==")
    fn = [r["frame_noise"] for r in rows if r["frame_noise"] != ""]
    gs = [r["gshift"] for r in rows]
    print(f"   sensor noise: median |frame-frame| {np.median(fn):.2f} luma "
          f"-> sigma ~ {np.median(fn)/0.954:.2f} (motion gate breaks near 8)")
    print(f"   global change |dmean|: median {np.median(gs):.2f}, "
          f"max {max(gs):.2f} (guard trips above {hmc.GLOBAL_CHANGE_MAX})\n")

    print("== palm detector, RAW (before any gate) ==")
    for t in SCORE_SWEEP:
        key = f"n{int(t*100):03d}"
        print(f"   frames with >=1 detection at score >= {t:.1f} : "
              f"{100*sum(1 for r in rows if r[key])/N:5.1f}%  "
              f"(median count {np.median([r[key] for r in rows]):.0f})")
    print(f"   best score: median {np.median(best):.3f}  "
          f"p90 {np.percentile(best, 90):.3f}  max {best.max():.3f}")

    print("\n== stage funnel (frames accepted at each stage, of all frames) ==")
    print(f"   detector fired (>= {hmc.MIN_SCORE})   : {pct('n050'):5.1f}%")
    print(f"   + size gate (>= {hmc.MIN_BOX_FRAC_REACQUIRE} of frame): "
          f"{pct('size_ok'):5.1f}%")
    print(f"   + motion gate                     : {pct('motion_ok'):5.1f}%")
    print(f"   + CNN verifier (>= {hmc.VERIFIER_THRESHOLD})        : "
          f"{pct('verifier_ok'):5.1f}%")
    susp = sum(1 for r in rows if r["gshift"] > hmc.GLOBAL_CHANGE_MAX)
    print(f"   frames where the global-change guard fires: {susp} "
          f"({100*susp/N:.1f}%)")

    mf = [r["m_frac"] for r in rows if r["m_frac"] != ""]
    if mf:
        print(f"\n   motion frac of the top candidate: median {np.median(mf):.4f} "
              f"(floor {hmc.MOTION_MIN})  zero-motion candidates: "
              f"{100*np.mean([m <= 0 for m in mf]):.1f}%")
    ph = [r["p_hand"] for r in rows if r["p_hand"] != ""]
    if ph:
        print(f"   verifier P(hand) of the top candidate: median {np.median(ph):.3f} "
              f"(threshold {hmc.VERIFIER_THRESHOLD})")

    if gt is None:
        print(f"\nper-frame log -> {out}")
        return

    # ---------------- attribution ---------------------------------------
    print("\n== ATTRIBUTION: what happens to frames where the hand IS there ==")
    print("   (MediaPipe hand present at conf 0.3; 'on the hand' = the detection's"
          "\n    box contains MediaPipe's palm centre, which is the fair test at"
          "\n    15 fps because MediaPipe's tracked centre lags during motion)")
    on = [r for r in rows if r["mp"]]
    M = len(on)
    print(f"   frames with a hand (MediaPipe)     : {M} of {N} ({100*M/N:.1f}%)\n")
    det = sum(1 for r in on if r["n050"] > 0)
    onh = sum(1 for r in on if r["hand_det"])
    print(f"   our detector fired AT ALL there    : {det:>4} ({100*det/M:5.1f}%)")
    print(f"   ...a detection CONTAINING the hand : {onh:>4} ({100*onh/M:5.1f}%)"
          f"   <- detector recall on the hand")
    hs = [r["hand_det_score"] for r in on if r["hand_det"]]
    if hs:
        print(f"      score of those detections      : median {np.median(hs):.3f}")
    # NESTED counts: each stage conditioned on the previous one.  Counting them
    # marginally (as this script first did) can print identical numbers for two
    # stages -- "motion 86, verifier 86" looked like a perfect verifier when only
    # 73 frames actually passed both.
    hsz = sum(1 for r in on if r["hand_size_ok"])
    hmot = sum(1 for r in on if r["hand_size_ok"] and r["hand_motion_ok"])
    hver = sum(1 for r in on if r["hand_size_ok"] and r["hand_motion_ok"]
               and r["hand_verifier_ok"])
    print(f"   ...of those, size gate passed      : {hsz:>4} "
          f"({100*hsz/M:5.1f}% of hand frames)")
    print(f"   ...of those, motion gate passed    : {hmot:>4} "
          f"({100*hmot/M:5.1f}% of hand frames)")
    print(f"   ...of those, verifier passed       : {hver:>4} "
          f"({100*hver/M:5.1f}% of hand frames)"
          f"   <- end to end on the real hand")
    hm = [r["hand_m_frac"] for r in on if r["hand_m_frac"] != ""]
    if hm:
        print(f"      motion frac on hand detections : median {np.median(hm):.4f}")
        print(f"      ...below the {hmc.MOTION_MIN} floor          : "
              f"{100*np.mean([m < hmc.MOTION_MIN for m in hm]):.1f}%")
    hp = [r["hand_p_hand"] for r in on if r["hand_p_hand"] != ""]
    if hp:
        print(f"      verifier P(hand) on hand dets  : median {np.median(hp):.3f}  "
              f"p90 {np.percentile(hp, 90):.3f}")
        print(f"      ...below the {hmc.VERIFIER_THRESHOLD} threshold        : "
              f"{100*np.mean([p < hmc.VERIFIER_THRESHOLD for p in hp]):.1f}%")

    print("\n== where the pipeline loses the hand (of hand frames) ==")
    no_fire = sum(1 for r in on if r["n050"] == 0)
    lost_det = M - onh
    lost_size = onh - hsz
    lost_mot = hsz - hmot
    lost_ver = hmot - hver
    print(f"   detector did not fire at all       : {no_fire:>4} "
          f"({100*no_fire/M:5.1f}%)")
    print(f"   detector fired, but off the hand   : {lost_det-no_fire:>4} "
          f"({100*(lost_det-no_fire)/M:5.1f}%)")
    print(f"   size gate rejected                 : {lost_size:>4} "
          f"({100*lost_size/M:5.1f}%)")
    print(f"   motion gate rejected               : {lost_mot:>4} "
          f"({100*lost_mot/M:5.1f}%)")
    print(f"   CNN verifier rejected              : {lost_ver:>4} "
          f"({100*lost_ver/M:5.1f}%)")
    print(f"   survived the whole chain           : {hver:>4} "
          f"({100*hver/M:5.1f}%)")

    offs = [r["off_norm"] for r in on if r["off_norm"] != ""]
    if offs:
        o = np.array(offs)
        print(f"\n== how far off are our detections when the hand is there? ==")
        print("   (box centre to MediaPipe palm centre, in units of the box width)")
        print(f"   median {np.median(o):.2f}  p25 {np.percentile(o,25):.2f}  "
              f"p75 {np.percentile(o,75):.2f}  p90 {np.percentile(o,90):.2f}")
        for t in (0.35, 0.5, 1.0, 2.0):
            print(f"   within {t:>4.2f} box widths: {100*np.mean(o <= t):5.1f}%"
                  f"  ({int((o <= t).sum())} of {len(o)} frames with a detection)")

    # Temporal coherence: if the detector were tracking the hand, successive
    # detections would form a smooth trajectory. If it is firing on assorted
    # scene features, the box centre jumps between unrelated places frame to
    # frame.  Compared against MediaPipe's own frame-to-frame step, which is what
    # a coherent hand trajectory looks like in this clip.
    def steps(key):
        pts, out_ = [], []
        for r in rows:
            v = r[key]
            if v == "":
                pts.append(None)
                continue
            pts.append(np.array(v, dtype=float))
        for i in range(1, len(pts)):
            if pts[i] is not None and pts[i - 1] is not None:
                out_.append(float(np.linalg.norm(pts[i] - pts[i - 1])))
        return np.array(out_) if out_ else np.array([np.nan])

    ours = steps("box_c")
    theirs = steps("mp_c")
    print(f"\n== temporal coherence (frame-to-frame movement, frame widths) ==")
    print(f"   our detection box centre : median step "
          f"{np.nanmedian(ours):.3f}  p90 {np.nanpercentile(ours, 90):.3f}")
    print(f"   MediaPipe palm centre    : median step "
          f"{np.nanmedian(theirs):.3f}  p90 {np.nanpercentile(theirs, 90):.3f}")
    if np.nanmedian(theirs) > 0:
        print(f"   ratio (ours/theirs)      : "
              f"{np.nanmedian(ours)/np.nanmedian(theirs):.1f}x  "
              f"<- a coherent track should be ~1x")

    if blob_verifier is not None:
        print("\n== can motion blobs recover what the detector misses? ==")
        det_fail = [r for r in on if not r["hand_size_ok"]]
        det_fail_any = [r for r in on if not r["hand_det"]]
        print(f"   hand frames the detector failed (no hand-containing box): "
              f"{len(det_fail_any)}")
        if det_fail_any:
            pr = 100 * np.mean([r["blob_proposed"] for r in det_fail_any])
            cf = 100 * np.mean([r["blob_confirmed"] for r in det_fail_any])
            print(f"      motion proposed the hand there : {pr:5.1f}%")
            print(f"      ...and the verifier confirmed  : {cf:5.1f}%")
        print(f"   hand frames where the detector found it but the chain rejected it: "
              f"{len([r for r in on if r['hand_size_ok'] and not r['hand_verifier_ok']])}")
        allf = [r for r in rows if r["mp"]]
        if allf:
            print(f"   across all {len(allf)} hand frames: blob proposed "
                  f"{100*np.mean([r['blob_proposed'] for r in allf]):5.1f}%, "
                  f"confirmed {100*np.mean([r['blob_confirmed'] for r in allf]):5.1f}%")
        print(f"   blob candidates per frame: median {np.median([r['n_blob_raw'] for r in rows]):.0f} "
              f"proposed, {np.median([r['n_blob_ok'] for r in rows]):.0f} confirmed")

    print(f"\nper-frame log -> {out}")

    print("\n== colour cast and noise vs the existing clips ==")
    print(f"   {'clip':<22}{'luma':>7}{'B':>7}{'G':>7}{'R':>7}{'R/B':>7}{'sigma':>7}")
    for c in [args.clip] + [d for d in DAYLIGHT if os.path.exists(os.path.join(HERE, d))]:
        s = clip_colour(os.path.join(HERE, c))
        print(f"   {c:<22}{s['luma']:>7.1f}{s['b']:>7.1f}{s['g']:>7.1f}"
              f"{s['r']:>7.1f}{s['r']/max(s['b'],1e-6):>7.3f}{s['sigma']:>7.2f}")

    if blob_verifier is not None:
        print("\n== can motion blobs recover what the detector misses? ==")
        det_fail = [r for r in on if not r["hand_size_ok"]]
        det_fail_any = [r for r in on if not r["hand_det"]]
        print(f"   hand frames the detector failed (no hand-containing box): "
              f"{len(det_fail_any)}")
        if det_fail_any:
            pr = 100 * np.mean([r["blob_proposed"] for r in det_fail_any])
            cf = 100 * np.mean([r["blob_confirmed"] for r in det_fail_any])
            print(f"      motion proposed the hand there : {pr:5.1f}%")
            print(f"      ...and the verifier confirmed  : {cf:5.1f}%")
        print(f"   hand frames where the detector found it but the chain rejected it: "
              f"{len([r for r in on if r['hand_size_ok'] and not r['hand_verifier_ok']])}")
        allf = [r for r in rows if r["mp"]]
        if allf:
            print(f"   across all {len(allf)} hand frames: blob proposed "
                  f"{100*np.mean([r['blob_proposed'] for r in allf]):5.1f}%, "
                  f"confirmed {100*np.mean([r['blob_confirmed'] for r in allf]):5.1f}%")
        print(f"   blob candidates per frame: median {np.median([r['n_blob_raw'] for r in rows]):.0f} "
              f"proposed, {np.median([r['n_blob_ok'] for r in rows]):.0f} confirmed")

    print(f"\nper-frame log -> {out}")


if __name__ == "__main__":
    main()
