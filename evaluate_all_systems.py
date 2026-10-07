"""Evaluate every system in one session, on the same footage, against the same truth.

This replaces running evaluate_pipelines.py once per configuration for the report's
main comparison, so that the dark columns and the light columns come from one session
rather than two. Frame timestamps are the deterministic ones from evaluate_pipelines.

Systems
  MediaPipe (raw)        mp.solutions.hands in tracking mode on the frame as recorded
  MediaPipe + CLAHE      same, on CLAHE-processed frames (L channel in LAB)
  MediaPipe + gamma      same, on luminance-gamma frames (gamma 0.5)
  Ours default           verifier_cnn.pt at 0.30, original detector, no motion blobs
  Ours big verifier      verifier_cnn_big.pt at 0.61, original detector, no blobs
  Dark v3                fine-tuned detector, big verifier at 0.61, motion blobs on,
                         acquisition agreement 3 passes at 0.05 frame widths
  Dark v4                Dark v3 plus the tracking hold, 3 candidates at 0.06

The truth is MediaPipe static mode on the RAW frames, borrowed within +-5 frames. It is
never computed on a preprocessed frame, so a preprocessing baseline is judged by the same
yardstick as everything else.

Outputs (all under results/)
  per_frame_<key>.csv        one row per clip per frame per system
  summary_all_systems.csv    aggregate metrics per system per clip
  summary_all_systems.txt    the same, as a table
  summary_vll_roles.csv      VeryLowLight metrics split by train / val / test blocks
  mp_sanity.txt              new raw MediaPipe runs against the cached ones

    .venv-blazepalm\\Scripts\\python.exe evaluate_all_systems.py
"""
import csv
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazpalm(githubmediapipe)", "BlazePalm", "ML")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                                    # noqa: E402
from crop_verifier import make_verifier                               # noqa: E402
from hand_mouse_cursor import (HandLocator, PlausibilityGate,         # noqa: E402
                               MotionGate, load_blob_verifier, load_verifier)
from evaluate_pipelines import (CLIPS, MATCH_FLOOR, clip_timestep,    # noqa: E402
                                matched, truth_at)
from vll_split import BLOCK, SPLIT_SEED, block_roles                  # noqa: E402

RES = os.path.join(HERE, "results")
MP_NPZ = os.path.join(RES, "mp_positions.npz")
VLL = "VeryLowLight.mp4"
HAND_CLIPS = {c for c, _g, _k, kd in CLIPS if kd == "hand"}
NOHAND_CLIPS = {c for c, _g, _k, kd in CLIPS if kd == "nohand"}

# system key -> (detector weights, verifier file, threshold, blobs, acquire, track)
LOCATOR_SYSTEMS = [
    ("ours_default", None, "verifier_cnn.pt", 0.30, False, 0, 0),
    ("ours_big61", None, "verifier_cnn_big.pt", 0.61, False, 0, 0),
    ("dark_v3", "palmdetector_dark.pth", "verifier_cnn_big.pt", 0.61, True, 3, 0),
    ("dark_v4", "palmdetector_dark.pth", "verifier_cnn_big.pt", 0.61, True, 3, 3),
]
ACQ_TOL, TRACK_TOL = 0.05, 0.06

MP_SYSTEMS = [("mp_raw", "raw"), ("mp_clahe", "clahe"), ("mp_gamma", "gamma")]

SYSTEM_LABELS = {
    "mp_raw": "MediaPipe (tracking, raw)",
    "mp_clahe": "MediaPipe + CLAHE",
    "mp_gamma": "MediaPipe + gamma 0.5",
    "ours_default": "Ours default (verifier_cnn.pt @0.30)",
    "ours_big61": "Ours big verifier (verifier_cnn_big.pt @0.61)",
    "dark_v3": "Dark v3 (acquire agreement)",
    "dark_v4": "Dark v4 (acquire + tracking hold)",
}


def load_detector(weights, cache={}):
    """Detector weights are either in the vendored port folder or in the project root."""
    key = weights or "palmdetector.pth"
    if key not in cache:
        name = key
        path = None
        for cand in (os.path.join(ML, name), os.path.join(HERE, name)):
            if os.path.isfile(cand):
                path = cand
                break
        if path is None:
            sys.exit(f"detector weights not found: {name} (looked in {ML} and {HERE})")
        det = PalmDetector()
        det.load_weights(path)
        det.load_anchors(os.path.join(ML, "anchors.npy"))
        det.eval()
        cache[key] = det
    return cache[key]


def gt_for(clip, gt_static):
    return gt_static


def evaluate_locator(system, det, verifier, thr, blobs, acquire, track,
                     clip, gt_static, blob_verifier, max_frames):
    """Replay one clip through one configuration, recording every frame."""
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    dt = clip_timestep(cap)
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=verifier,
                      verifier_threshold=thr, motion_gate=MotionGate(),
                      use_motion_blobs=blobs, blob_verifier=blob_verifier,
                      acquire_consensus=acquire, acquire_consensus_tol=ACQ_TOL,
                      track_consensus=track, track_consensus_tol=TRACK_TOL)
    rows = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or n >= max_frames:
            break
        n += 1
        mode_before = loc.mode
        pos, info = loc.update(frame, (n - 1) * dt, dt)
        hands = truth_at(gt_static, None, n)
        fw = frame.shape[1]
        if pos is None:
            rows.append(dict(clip=clip, frame=n, truth_present=bool(hands),
                             accepted=False, correct=False, dist_frac="",
                             mode_before=mode_before, p_hand=""))
        else:
            ok = matched(pos, hands, fw)
            d = min((float(np.hypot(pos[0] - hx, pos[1] - hy)) / fw
                     for hx, hy, _pw in hands), default=float("nan"))
            rows.append(dict(clip=clip, frame=n, truth_present=bool(hands),
                             accepted=True, correct=bool(ok),
                             dist_frac=("" if d != d else round(d, 6)),
                             mode_before=mode_before,
                             p_hand=("" if info.get("p_hand") is None
                                     else round(float(info["p_hand"]), 6))))
    cap.release()
    return rows


def evaluate_mp(cond, clip, positions, gt_static, max_frames):
    """One MediaPipe condition, matched against the raw static truth."""
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    cap.release()
    by_frame = {int(r[0]): r for r in positions}
    n_frames = int(max(by_frame)) if by_frame else 0
    rows = []
    for n in range(1, min(n_frames, max_frames) + 1):
        r = by_frame.get(n)
        hands = truth_at(gt_static, None, n)
        present = bool(r is not None and r[1] > 0.5)
        if not present:
            rows.append(dict(clip=clip, frame=n, truth_present=bool(hands),
                             accepted=False, correct=False, dist_frac="",
                             mode_before="", p_hand=""))
            continue
        pos = (float(r[2]), float(r[3]))
        ok = matched(pos, hands, fw)
        d = min((float(np.hypot(pos[0] - hx, pos[1] - hy)) / fw
                 for hx, hy, _pw in hands), default=float("nan"))
        rows.append(dict(clip=clip, frame=n, truth_present=bool(hands),
                         accepted=True, correct=bool(ok),
                         dist_frac=("" if d != d else round(d, 6)),
                         mode_before="", p_hand=""))
    return rows


def summarise(rows, role_mask=None):
    if role_mask is not None:
        rows = [r for r in rows if role_mask[r["frame"]]]
    frames = len(rows)
    truth = sum(1 for r in rows if r["truth_present"])
    acc = [r for r in rows if r["accepted"]]
    tp = sum(1 for r in acc if r["correct"])
    fp = len(acc) - tp
    prec = tp / len(acc) if acc else float("nan")
    rec = tp / truth if truth else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if acc and truth and (prec + rec) > 0 else float("nan")
    return dict(frames=frames, truth_frames=truth, accepted=len(acc), tp=tp, fp=fp,
                precision=prec, recall=rec, f1=f1,
                acceptance_rate=len(acc) / frames if frames else float("nan"))


def fmt(v, pct=True):
    if v != v:
        return ""
    return f"{100*v:.1f}" if pct else f"{v:.4f}"


def main():
    t_start = time.time()
    os.makedirs(RES, exist_ok=True)
    max_frames = 100000
    if "--frames" in sys.argv:
        max_frames = int(sys.argv[sys.argv.index("--frames") + 1])

    if not os.path.isfile(MP_NPZ):
        sys.exit("results/mp_positions.npz is missing. Run make_mp_positions.py with "
                 "the mediapipe venv first.")
    mpz = np.load(MP_NPZ)

    blob_verifier, _bp = load_blob_verifier(None)
    verifiers = {}
    for _k, _w, vf, _t, _b, _a, _tr in LOCATOR_SYSTEMS:
        if vf not in verifiers:
            verifiers[vf] = make_verifier(os.path.join(HERE, vf))
    detectors = {}
    for key, w, _vf, _t, _b, _a, _tr in LOCATOR_SYSTEMS:
        if (w or "palmdetector.pth") not in detectors:
            detectors[w or "palmdetector.pth"] = load_detector(w)

    per_frame = {}
    summary = []

    # ---------------------------------------------------------------- locators
    for clip, gtf, key, kind in CLIPS:
        z = np.load(os.path.join(HERE, gtf))
        gt_static = z[key]
        for syskey, w, vf, thr, blobs, acq, trk in LOCATOR_SYSTEMS:
            rows = evaluate_locator(syskey, detectors[w or "palmdetector.pth"],
                                    verifiers[vf], thr, blobs, acq, trk,
                                    clip, gt_static, blob_verifier, max_frames)
            per_frame.setdefault(syskey, []).extend(rows)
            s = summarise(rows)
            summary.append(dict(system=syskey, clip=clip, kind=kind, **s))
            print(f"  {syskey:<14} {clip:<22} accepted {s['accepted']:>4} "
                  f"F1 {fmt(s['f1'])}", flush=True)

    # ------------------------------------------------------------- mediapipe
    for clip, gtf, key, kind in CLIPS:
        z = np.load(os.path.join(HERE, gtf))
        gt_static = z[key]
        for syskey, cond in MP_SYSTEMS:
            arr = mpz[f"{cond}_track_{clip}"]
            rows = evaluate_mp(cond, clip, arr, gt_static, max_frames)
            per_frame.setdefault(syskey, []).extend(rows)
            s = summarise(rows)
            summary.append(dict(system=syskey, clip=clip, kind=kind, **s))
            print(f"  {syskey:<14} {clip:<22} accepted {s['accepted']:>4} "
                  f"F1 {fmt(s['f1'])}", flush=True)

    # ---------------------------------------------------------------- outputs
    for syskey, rows in per_frame.items():
        path = os.path.join(RES, f"per_frame_{syskey}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=["clip", "frame", "truth_present",
                                                "accepted", "correct", "dist_frac",
                                                "mode_before", "p_hand"])
            wtr.writeheader()
            for r in sorted(rows, key=lambda r: (r["clip"], r["frame"])):
                wtr.writerow(r)
        print(f"wrote {path} ({len(rows)} rows)")

    fields = ["system", "clip", "kind", "frames", "truth_frames", "accepted", "tp", "fp",
              "precision", "recall", "f1", "acceptance_rate"]
    with open(os.path.join(RES, "summary_all_systems.csv"), "w", newline="",
              encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for r in summary:
            wtr.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in r.items() if k in fields})

    # aggregate over hand clips, plus false locks on the no-hand clips
    lines = []
    lines.append("PER-CLIP AND AGGREGATE METRICS, one session, identical truth")
    lines.append("truth: MediaPipe static mode on raw frames, +-5 frame borrow")
    lines.append("")
    hdr = (f"{'system':<44}{'clip':<22}{'acc':>6}{'TP':>5}{'FP':>5}"
           f"{'prec':>7}{'rec':>7}{'F1':>7}{'locks':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for syskey in SYSTEM_LABELS:
        for r in summary:
            if r["system"] != syskey:
                continue
            locks = fmt(r["acceptance_rate"]) if r["kind"] == "nohand" else "-"
            lines.append(f"{SYSTEM_LABELS[syskey]:<44}{r['clip']:<22}"
                         f"{r['accepted']:>6}{r['tp']:>5}{r['fp']:>5}"
                         f"{fmt(r['precision']):>7}{fmt(r['recall']):>7}"
                         f"{fmt(r['f1']):>7}{locks:>7}")
        agg = summarise([x for x in per_frame[syskey] if x["clip"] in HAND_CLIPS])
        nh = [x for x in per_frame[syskey] if x["clip"] in NOHAND_CLIPS]
        locks = sum(1 for x in nh if x["accepted"]) / len(nh) if nh else float("nan")
        lines.append(f"{'  -> aggregate (hand clips)':<44}{'':<22}"
                     f"{agg['accepted']:>6}{agg['tp']:>5}{agg['fp']:>5}"
                     f"{fmt(agg['precision']):>7}{fmt(agg['recall']):>7}"
                     f"{fmt(agg['f1']):>7}{fmt(locks):>7}")
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(RES, "summary_all_systems.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)

    # ------------------------------------------------- VeryLowLight by role
    roles = block_roles(SPLIT_SEED)
    role_of = {n: roles[(n - 1) // BLOCK] for n in range(1, 652)}
    vll_rows = []
    for syskey, rows in per_frame.items():
        vrows = [r for r in rows if r["clip"] == VLL]
        for role in ("train", "val", "test"):
            s = summarise(vrows, role_mask={n: role_of.get(n) == role
                                            for n in range(1, 2000)})
            vll_rows.append(dict(system=syskey, role=role, **s))
    with open(os.path.join(RES, "summary_vll_roles.csv"), "w", newline="",
              encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=["system", "role", "frames", "truth_frames",
                                           "accepted", "tp", "fp", "precision",
                                           "recall", "f1", "acceptance_rate"])
        wtr.writeheader()
        for r in vll_rows:
            wtr.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in r.items()})

    print("\nVERYLOWLIGHT BY SPLIT ROLE (seed %d)" % SPLIT_SEED)
    print(f"{'system':<44}{'role':<7}{'frames':>7}{'truth':>7}{'acc':>6}{'F1':>7}")
    for r in vll_rows:
        print(f"{SYSTEM_LABELS[r['system']]:<44}{r['role']:<7}{r['frames']:>7}"
              f"{r['truth_frames']:>7}{r['accepted']:>6}{fmt(r['f1']):>7}")

    print(f"\ntotal {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
