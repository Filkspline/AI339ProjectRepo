"""Pipeline-level evaluation across the training seeds.

Two phases.

Phase 1, threshold selection. The operating threshold is treated as one hyperparameter
per variant, chosen once on the VeryLowLight validation blocks at the pipeline level,
using seed 0's checkpoint. The seed affects the weights; letting each seed also pick its
own threshold would fold selection noise into the seed variance.

Phase 2, per-seed evaluation. Each seed's checkpoint is run through the full pipeline on
all eight clips at the variant's selected threshold, recording per-clip and aggregate
precision, recall, F1 and no-hand false-lock rate, plus per-frame outcomes for the paired
tests.

Variants and the pipeline configuration each represents:
  verifier_cnn.pt          default      original detector, no motion blobs, no agreement rules
  verifier_cnn_big.pt      low-light    same, 130k-crop verifier
  verifier_cnn_vll.pt      vll-tuned    same, target-condition verifier
  verifier_dark.pt         dark-aug     same, dark-augmented verifier
  detector_seed*           dark         fine-tuned detector seed, big verifier, blobs on,
                                        both agreement rules on

    .venv-blazepalm\\Scripts\\python.exe seeds_pipeline_eval.py --phase select
    .venv-blazepalm\\Scripts\\python.exe seeds_pipeline_eval.py --phase eval
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazpalm(githubmediapipe)", "BlazePalm", "ML")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                                    # noqa: E402
from crop_verifier import SmallCNN, make_verifier                     # noqa: E402
from hand_mouse_cursor import (HandLocator, PlausibilityGate,         # noqa: E402
                               MotionGate, load_blob_verifier)
from evaluate_pipelines import (CLIPS, clip_timestep, matched,        # noqa: E402
                                truth_at)
from vll_split import BLOCK, SPLIT_SEED, block_roles                  # noqa: E402

RES = os.path.join(HERE, "results")
SEEDDIR = os.path.join(RES, "verifier_seeds")
DETDIR = os.path.join(RES, "detector_seeds")
THR_FILE = os.path.join(RES, "thresholds.json")
VLL = "VeryLowLight.mp4"
HAND_CLIPS = {c for c, _g, _k, kd in CLIPS if kd == "hand"}
NOHAND_CLIPS = {c for c, _g, _k, kd in CLIPS if kd == "nohand"}
GRID = [round(0.05 + 0.05 * i, 2) for i in range(19)]     # 0.05 .. 0.95

VARIANTS = ["verifier_cnn.pt", "verifier_cnn_big.pt", "verifier_cnn_vll.pt",
            "verifier_dark.pt"]
DARK_CONFIG = dict(blobs=True, acquire=3, track=3)
ACQ_TOL, TRACK_TOL = 0.05, 0.06


def roles_mask(role):
    roles = block_roles(SPLIT_SEED)
    return {n: roles[(n - 1) // BLOCK] == role for n in range(1, 652)}


def load_detector(name):
    for cand in (os.path.join(ML, name), os.path.join(HERE, name),
                 os.path.join(DETDIR, name)):
        if os.path.isfile(cand):
            det = PalmDetector()
            det.load_weights(cand)
            det.load_anchors(os.path.join(ML, "anchors.npy"))
            det.eval()
            return det
    sys.exit(f"detector not found: {name}")


def run_clip(det, verifier, thr, blobs, acquire, track, clip, gt_static,
             max_frames=100000, record=True):
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    dt = clip_timestep(cap)
    loc = HandLocator(det, gate=PlausibilityGate(), verifier=verifier,
                      verifier_threshold=thr, motion_gate=MotionGate(),
                      use_motion_blobs=blobs, blob_verifier=BLOBV,
                      acquire_consensus=acquire, acquire_consensus_tol=ACQ_TOL,
                      track_consensus=track, track_consensus_tol=TRACK_TOL)
    rows = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or n >= max_frames:
            break
        n += 1
        pos, _info = loc.update(frame, (n - 1) * dt, dt)
        hands = truth_at(gt_static, None, n)
        fw = frame.shape[1]
        if pos is None:
            rows.append((n, bool(hands), False, False, ""))
        else:
            ok = matched(pos, hands, fw)
            d = min((float(np.hypot(pos[0] - hx, pos[1] - hy)) / fw
                     for hx, hy, _pw in hands), default=float("nan"))
            rows.append((n, bool(hands), True, bool(ok),
                         "" if d != d else round(d, 6)))
    cap.release()
    return rows


def metrics(rows):
    frames = len(rows)
    truth = sum(1 for r in rows if r[1])
    acc = [r for r in rows if r[2]]
    tp = sum(1 for r in acc if r[3])
    fp = len(acc) - tp
    prec = tp / len(acc) if acc else float("nan")
    rec = tp / truth if truth else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if acc and truth and (prec + rec) else float("nan")
    return dict(frames=frames, truth_frames=truth, accepted=len(acc), tp=tp, fp=fp,
                precision=prec, recall=rec, f1=f1,
                acceptance_rate=len(acc) / frames if frames else float("nan"))


BLOBV = None


def gt_of(clip, gtf, key):
    z = np.load(os.path.join(HERE, gtf))
    return z[key]


def phase_select(max_frames):
    """Threshold per variant, chosen on the validation blocks of the target clip."""
    out = {}
    gtf, key = next((g, k) for c, g, k, _kd in CLIPS if c == VLL)
    gt = gt_of(VLL, gtf, key)
    vm = roles_mask("val")
    for name in VARIANTS:
        ckpt = os.path.join(SEEDDIR, f"{name}_seed0.pt")
        if not os.path.isfile(ckpt):
            print(f"  {name}: no seed-0 checkpoint yet, skipping")
            continue
        verifier = make_verifier(ckpt)
        det = load_detector("palmdetector.pth")
        best = (None, -1.0)
        curve = []
        for thr in GRID:
            rows = run_clip(det, verifier, thr, False, 0, 0, VLL, gt, max_frames)
            rows = [r for r in rows if vm.get(r[0], False)]
            m = metrics(rows)
            curve.append(dict(threshold=thr, f1=round(m["f1"], 6),
                              precision=round(m["precision"], 6),
                              recall=round(m["recall"], 6), accepted=m["accepted"]))
            if m["f1"] == m["f1"] and m["f1"] > best[1]:
                best = (thr, m["f1"])
            print(f"  {name} thr {thr:.2f}  val F1 {m['f1']:.3f}", flush=True)
        out[name] = dict(threshold=best[0], val_f1=round(best[1], 6), curve=curve,
                         selected_on=f"VLL validation blocks, split seed {SPLIT_SEED}")
        print(f"  -> {name}: threshold {best[0]} (val F1 {best[1]:.3f})", flush=True)
    # The dark pipeline runs a fine-tuned detector, so its verifier threshold has
    # to be selected against that detector rather than inherited from a selection
    # made with the original one. Selected once, with seed 0, on the same
    # validation blocks, and then applied to every detector seed (the same
    # convention the verifier variants use).
    if "verifier_cnn_big.pt" in out:
        ckpt = os.path.join(SEEDDIR, "verifier_cnn_big.pt_seed0.pt")
        det_name = ("detector_seed0.pth"
                    if os.path.isfile(os.path.join(DETDIR, "detector_seed0.pth"))
                    else "palmdetector_dark.pth")
        verifier = make_verifier(ckpt)
        det = load_detector(det_name)
        best = (None, -1.0)
        curve = []
        for thr in GRID:
            rows = run_clip(det, verifier, thr, clip=VLL, gt_static=gt,
                            max_frames=max_frames, **DARK_CONFIG)
            rows = [r for r in rows if vm.get(r[0], False)]
            m = metrics(rows)
            curve.append(dict(threshold=thr, f1=round(m["f1"], 6),
                              precision=round(m["precision"], 6),
                              recall=round(m["recall"], 6), accepted=m["accepted"]))
            if m["f1"] == m["f1"] and m["f1"] > best[1]:
                best = (thr, m["f1"])
            print(f"  dark_pipeline ({det_name}) thr {thr:.2f}  val F1 {m['f1']:.3f}",
                  flush=True)
        out["dark_pipeline"] = dict(
            threshold=best[0], val_f1=round(best[1], 6), curve=curve,
            detector=det_name, verifier="verifier_cnn_big.pt_seed0.pt",
            selected_on=f"VLL validation blocks, split seed {SPLIT_SEED}, "
                        f"dark config (blobs on, consensus 3/3)")
        print(f"  -> dark_pipeline: threshold {best[0]} (val F1 {best[1]:.3f})",
              flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(THR_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {THR_FILE}")


def phase_eval(max_frames):
    global BLOBV
    BLOBV, _p = load_blob_verifier(None)
    with open(THR_FILE, encoding="utf-8") as f:
        thr_map = json.load(f)
    per_frame_dir = os.path.join(RES, "seed_per_frame")
    os.makedirs(per_frame_dir, exist_ok=True)

    # --- verifier variants, original detector ---
    for name in VARIANTS:
        if name not in thr_map:
            continue
        thr = thr_map[name]["threshold"]
        seeds = sorted(int(f.split("seed")[1].split(".")[0])
                       for f in os.listdir(SEEDDIR)
                       if f.startswith(f"{name}_seed") and f.endswith(".pt"))
        print(f"\n=== {name} @ {thr}  ({len(seeds)} seeds) ===", flush=True)
        rows_all = []
        for seed in seeds:
            ckpt = os.path.join(SEEDDIR, f"{name}_seed{seed}.pt")
            verifier = make_verifier(ckpt)
            det = load_detector("palmdetector.pth")
            for clip, gtf, key, kind in CLIPS:
                gt = gt_of(clip, gtf, key)
                rows = run_clip(det, verifier, thr, False, 0, 0, clip, gt, max_frames)
                m = metrics(rows)
                rows_all.append(dict(variant=name, seed=seed, clip=clip, kind=kind,
                                     threshold=thr, **m))
                with open(os.path.join(per_frame_dir,
                                       f"{name}_seed{seed}_{clip.replace('.mp4','')}.csv"),
                          "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["frame", "truth_present", "accepted", "correct",
                                "dist_frac"])
                    w.writerows(rows)
            print(f"  seed {seed} done", flush=True)
        write_variant(rows_all, f"seed_eval_{name}")

    # --- detector seeds, dark pipeline ---
    if os.path.isdir(DETDIR) and "dark_pipeline" in thr_map:
        thr = thr_map["dark_pipeline"]["threshold"]
        seeds = sorted(int(f.split("seed")[1].split(".")[0])
                       for f in os.listdir(DETDIR)
                       if f.startswith("detector_seed") and f.endswith(".pth"))
        print(f"\n=== detector fine-tune seeds @ {thr}  ({len(seeds)} seeds) ===",
              flush=True)
        verifier = make_verifier(os.path.join(SEEDDIR, "verifier_cnn_big.pt_seed0.pt"))
        rows_all = []
        for seed in seeds:
            det = load_detector(f"detector_seed{seed}.pth")
            for clip, gtf, key, kind in CLIPS:
                gt = gt_of(clip, gtf, key)
                rows = run_clip(det, verifier, thr, **DARK_CONFIG, clip=clip,
                                gt_static=gt, max_frames=max_frames)
                m = metrics(rows)
                rows_all.append(dict(variant="detector_seed", seed=seed, clip=clip,
                                     kind=kind, threshold=thr, **m))
                with open(os.path.join(per_frame_dir,
                                       f"detector_seed{seed}_{clip.replace('.mp4','')}.csv"),
                          "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["frame", "truth_present", "accepted", "correct",
                                "dist_frac"])
                    w.writerows(rows)
            print(f"  detector seed {seed} done", flush=True)
        write_variant(rows_all, "seed_eval_detector_seed")


def write_variant(rows_all, stem):
    fields = ["variant", "seed", "clip", "kind", "threshold", "frames", "truth_frames",
              "accepted", "tp", "fp", "precision", "recall", "f1", "acceptance_rate"]
    path = os.path.join(RES, f"{stem}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_all:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})
    # per-seed aggregates, so the SD across seeds can be taken
    seeds = sorted({r["seed"] for r in rows_all})
    agg = []
    for seed in seeds:
        rs = [r for r in rows_all if r["seed"] == seed]
        hand = [r for r in rs if r["clip"] in HAND_CLIPS]
        nh = [r for r in rs if r["clip"] in NOHAND_CLIPS]
        tp = sum(r["tp"] for r in hand)
        fp = sum(r["fp"] for r in hand)
        truth = sum(r["truth_frames"] for r in hand)
        accepted = sum(r["accepted"] for r in hand)
        frames = sum(r["frames"] for r in hand)
        prec = tp / accepted if accepted else float("nan")
        rec = tp / truth if truth else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if accepted and truth and (prec + rec) else float("nan")
        nh_acc = sum(r["accepted"] for r in nh)
        nh_fr = sum(r["frames"] for r in nh)
        agg.append(dict(seed=seed, agg_f1=f1, agg_precision=prec, agg_recall=rec,
                        agg_accepted=accepted, agg_frames=frames,
                        false_lock_rate=nh_acc / nh_fr if nh_fr else float("nan")))
    apath = os.path.join(RES, f"{stem}_aggregate.csv")
    with open(apath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"wrote {path}")
    print(f"wrote {apath}")
    txt = [f"{stem}: {len(seeds)} seeds"]
    for k in ("agg_f1", "agg_precision", "agg_recall", "false_lock_rate"):
        v = np.array([a[k] for a in agg], dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            txt.append(f"  {k:<18} mean {v.mean():.4f}  SD {v.std(ddof=1):.4f}"
                       f"  min {v.min():.4f}  max {v.max():.4f}")
    print("\n".join(txt), flush=True)
    with open(os.path.join(RES, f"{stem}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["select", "eval"], required=True)
    ap.add_argument("--frames", type=int, default=100000)
    args = ap.parse_args()
    if args.phase == "select":
        phase_select(args.frames)
    else:
        phase_eval(args.frames)


if __name__ == "__main__":
    main()
