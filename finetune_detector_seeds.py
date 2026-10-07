"""Multi-seed detector fine-tune, with the new train / val / test block split.

Same model surgery, loss, optimiser and augmentation as finetune_detector_dark.py, which
this imports rather than copies. What changes is which frames are visible at each stage:

  train  VeryLowLight frames in the training blocks
  val    VeryLowLight frames in the validation blocks. The checkpoint is the epoch with
         the lowest validation loss, computed on a fixed subsample of these frames with
         no augmentation. The validation containment rate is also recorded.
  test   VeryLowLight frames in the test blocks. Evaluated once per seed, at the end.

The seven light clips are reported too, as an out-of-condition check, and are never
trained on here.

The collapse guard from the original script is kept: a model firing above 0.5 on more
than 100 anchors of a light frame is refused rather than saved. Trips are counted and
reported.

    .venv-blazepalm\\Scripts\\python.exe finetune_detector_seeds.py --seeds 0,...,9

Outputs under results/detector_seeds/:
  detector_seed<seed>.pth, detector_seed<seed>.json
  detector_seeds_curves.csv        per-epoch train/val loss, mean and SD across seeds
  detector_seeds_summary.csv/.txt  per-seed and aggregate results
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazpalm(githubmediapipe)", "BlazePalm", "ML")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                                       # noqa: E402
from hand_pipeline import box_to_pixels, letterbox                       # noqa: E402
from build_dark_dataset import make_variant                              # noqa: E402
from build_vll_dataset import target_stats                               # noqa: E402
from finetune_detector_dark import (LIGHT_CLIPS, gt_targets,             # noqa: E402
                                    match_anchors, to_tensor)
from vll_split import SPLIT_SEED, block_roles                            # noqa: E402
from evaluate_pipelines import clip_timestep                             # noqa: E402

OUTDIR = os.path.join(HERE, "results", "detector_seeds")
CLIP = "VeryLowLight.mp4"
GT = "gt_verylowlight.npz"
GT_KEY = "static_conf0.3"
LM_KEY = "static_conf0.3_landmarks"
COLLAPSE_HOT = 100          # refuse to save above this many anchors >= 0.5


def role_map():
    """Frame index -> role, for the frames that exist in the clip."""
    roles = block_roles(SPLIT_SEED)
    return {n: roles[(n - 1) // 50] for n in range(1, 652)}


def load_frames():
    """Frames with a hand and landmarks, grouped by the new split role."""
    z = np.load(os.path.join(HERE, GT))
    gt, lm_all = z[GT_KEY], z[LM_KEY]
    roles = role_map()
    pools = {"train": [], "val": [], "test": []}
    cap = cv2.VideoCapture(os.path.join(HERE, CLIP))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        row = gt[gt[:, 0] == n]
        if not (len(row) and row[0, 1] and not np.isnan(lm_all[n - 1]).any()):
            continue
        pools[roles.get(n, "test")].append((n, frame.copy()))
    cap.release()
    return gt, lm_all, pools


def sample_targets(gt, lm_all, n_, frame, det, augment_rng=None, t_mean=None,
                   t_std=None, light_frac=0.25):
    row = gt[gt[:, 0] == n_]
    fw, fh = frame.shape[1], frame.shape[0]
    cx, cy, pw = float(row[0, 2]), float(row[0, 3]), float(row[0, 4])
    box = (cx - 1.3 * pw, cy - 1.3 * pw, cx + 1.3 * pw, cy + 1.3 * pw)
    L = lm_all[n_ - 1]
    kps = ((float(L[0, 0]) * fw, float(L[0, 1]) * fh),
           (float(L[9, 0]) * fw, float(L[9, 1]) * fh))
    img = frame
    if augment_rng is not None and augment_rng.random() > light_frac:
        img, _s, _b = make_variant(frame, t_mean, t_std, augment_rng)
    padded, scale, left, top = letterbox(img, 256)
    tgt, mask = gt_targets(box, kps, det.anchors, scale, left, top)
    idx = match_anchors(box, scale, left, top, det.anchors)
    if len(idx) == 0:
        return None
    tgt_pos = np.zeros(det.anchors.shape[0], dtype=np.float32)
    tgt_pos[idx] = 1.0
    return padded, tgt, mask, tgt_pos


def batch_from(det, gt, lm_all, items, rng=None, t_mean=None, t_std=None):
    cb, rb, mb, tb = [], [], [], []
    for n_, frame in items:
        out = sample_targets(gt, lm_all, n_, frame, det, rng, t_mean, t_std)
        if out is None:
            continue
        padded, tgt, mask, tgt_pos = out
        cb.append(padded)
        rb.append(tgt)
        mb.append(mask)
        tb.append(tgt_pos)
    if not cb:
        return None
    return (torch.stack([to_tensor(p) for p in cb]),
            torch.from_numpy(np.stack(tb)),
            torch.from_numpy(np.stack(rb)),
            torch.from_numpy(np.stack(mb)))


def loss_terms(det, x, tgt, raw, mask, bce, pos_weight, reg_weight):
    c, r = det(x)
    cls = bce(c[:, :, 0], tgt) * (1.0 + (pos_weight - 1.0) * tgt)
    reg_m = mask * tgt.unsqueeze(-1)
    diff = F.smooth_l1_loss(r, raw, reduction="none") * reg_m
    return cls.mean() + reg_weight * (diff.sum() / reg_m.sum().clamp(min=1.0))


def val_loss(det, gt, lm_all, val_frames, bce, args, t_mean, t_std):
    """Validation loss on a fixed subsample of the validation frames, no augmentation."""
    sub = val_frames[:args.val_loss_frames]
    tot, n = 0.0, 0
    det.eval()
    with torch.no_grad():
        for i in range(0, len(sub), args.batch):
            b = batch_from(det, gt, lm_all, sub[i:i + args.batch])
            if b is None:
                continue
            x, tgt, raw, mask = b
            tot += float(loss_terms(det, x, tgt, raw, mask, bce, args.pos_weight,
                                    args.reg_weight))
            n += 1
    return tot / max(n, 1)


def containment(det, gt, frames, tol=0.15):
    """Box-contains-hand rate on a frame list, as in the original script."""
    hit = fired = 0
    det.eval()
    with torch.no_grad():
        for n_, frame in frames:
            row = gt[gt[:, 0] == n_]
            cx, cy = float(row[0, 2]), float(row[0, 3])
            h, w = frame.shape[:2]
            padded, scale, left, top = letterbox(frame, 256)
            c, r = det(to_tensor(padded).unsqueeze(0))
            boxes = det._decode_boxes(r[0], det.anchors)
            scores = c[0, :, 0].sigmoid()
            k = int(torch.argmax(scores))
            if float(scores[k]) < 0.5:
                continue
            fired += 1
            b = boxes[k]
            bx0 = (float(b[0]) * 256.0 - left) / scale
            by0 = (float(b[1]) * 256.0 - top) / scale
            bx1 = (float(b[2]) * 256.0 - left) / scale
            by1 = (float(b[3]) * 256.0 - top) / scale
            if (bx0 - tol * w <= cx <= bx1 + tol * w
                    and by0 - tol * h <= cy <= by1 + tol * h):
                hit += 1
    return hit / max(len(frames), 1), fired / max(len(frames), 1)


def hot_anchors(det):
    """Collapse guard probe: anchors scoring above 0.5 on a light frame."""
    cap = cv2.VideoCapture(os.path.join(HERE, "CloseLightMoving.mp4"))
    ok, pf = cap.read()
    cap.release()
    if not ok:
        return -1
    padded, _s, _l, _t = letterbox(pf, 256)
    with torch.no_grad():
        c, _r = det(to_tensor(padded).unsqueeze(0))
    return int((c[0, :, 0].sigmoid() >= 0.5).sum())


def clip_frames(clip, gt):
    """Frames where the truth has a hand, for the out-of-condition clips."""
    z = np.load(os.path.join(HERE, "gt_clips.npz"))
    g = z[f"static:{clip}"]
    frames = []
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        row = g[g[:, 0] == n]
        if len(row) and row[0, 1]:
            frames.append((n, frame.copy()))
    cap.release()
    return g, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--light-frac", type=float, default=0.25)
    ap.add_argument("--val-loss-frames", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(OUTDIR, exist_ok=True)

    gt, lm_all, pools = load_frames()
    print(f"frames: train {len(pools['train'])}, val {len(pools['val'])}, "
          f"test {len(pools['test'])}")
    if not pools["train"] or not pools["val"]:
        sys.exit("empty train or validation pool")

    t_mean, t_std = target_stats(CLIP)[:2]
    light = {c: clip_frames(c, gt) for c in LIGHT_CLIPS}

    rows, curves = [], {}
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(6)
        det = PalmDetector()
        det.load_weights(os.path.join(ML, "palmdetector.pth"))
        det.load_anchors(os.path.join(ML, "anchors.npy"))
        for p in det.parameters():
            p.requires_grad = False
        for nm in ("class_32", "class_16", "class_8", "reg_32", "reg_16", "reg_8"):
            for p in getattr(det, nm).parameters():
                p.requires_grad = True
        params = [p for p in det.parameters() if p.requires_grad]
        n_tr = sum(p.numel() for p in params)
        opt = torch.optim.Adam(params, lr=args.lr)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        best_loss, best_state, best_ep = float("inf"), None, 0
        steps = max(1, len(pools["train"]) // args.batch)
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            det.train()
            tot, nc = 0.0, 0
            for _ in range(steps):
                items = [pools["train"][int(rng.integers(len(pools["train"])))]
                         for _ in range(args.batch)]
                b = batch_from(det, gt, lm_all, items, rng, t_mean, t_std)
                if b is None:
                    continue
                x, tgt, raw, mask = b
                opt.zero_grad()
                loss = loss_terms(det, x, tgt, raw, mask, bce, args.pos_weight,
                                  args.reg_weight)
                loss.backward()
                opt.step()
                tot += float(loss)
                nc += 1
            tr_loss = tot / max(nc, 1)
            va_loss = val_loss(det, gt, lm_all, pools["val"], bce, args, t_mean, t_std)
            va_cont = float("nan")
            if ep % args.eval_every == 0 or ep == args.epochs:
                va_cont, _fire = containment(det, gt, pools["val"])
            curves.setdefault(ep, []).append((tr_loss, va_loss, va_cont))
            print(f"  epoch {ep:>2}/{args.epochs}  train {tr_loss:.4f}  "
                  f"val {va_loss:.4f}  val-contain "
                  f"{'' if va_cont != va_cont else f'{100*va_cont:.1f}%'}", flush=True)
            if va_loss < best_loss:
                best_loss, best_ep = va_loss, ep
                best_state = {k: v.clone() for k, v in det.state_dict().items()}
        train_seconds = time.time() - t0

        det.load_state_dict(best_state)
        n_hot = hot_anchors(det)
        tripped = n_hot > COLLAPSE_HOT
        va_cont, va_fire = containment(det, gt, pools["val"])
        te_cont, te_fire = containment(det, gt, pools["test"])
        light_cont = {}
        for c in LIGHT_CLIPS:
            g, fr = light[c]
            light_cont[c], _f = containment(det, g, fr)
        light_mean = float(np.mean(list(light_cont.values())))

        path = os.path.join(OUTDIR, f"detector_seed{seed}.pth")
        if tripped:
            print(f"  COLLAPSE GUARD TRIPPED: {n_hot} anchors >= 0.5, not saved")
        else:
            torch.save(det.state_dict(), path)
        row = dict(seed=seed, trainable_params=n_tr, train_seconds=round(train_seconds, 1),
                   best_epoch=best_ep, best_val_loss=round(best_loss, 6),
                   val_contain=round(va_cont, 6), val_fire=round(va_fire, 6),
                   test_contain=round(te_cont, 6), test_fire=round(te_fire, 6),
                   light_mean_contain=round(light_mean, 6),
                   collapse_hot_anchors=n_hot, collapse_tripped=bool(tripped),
                   saved=path if not tripped else "")
        for c, v in light_cont.items():
            row[f"contain_{c.replace('.mp4', '')}"] = round(v, 6)
        rows.append(row)
        with open(os.path.join(OUTDIR, f"detector_seed{seed}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(f"  saved={not tripped}  best epoch {best_ep}  "
              f"val-contain {100*va_cont:.1f}%  test-contain {100*te_cont:.1f}%  "
              f"light-mean {100*light_mean:.1f}%  ({train_seconds:.0f}s)", flush=True)

    # curves
    with open(os.path.join(HERE, "results", "detector_seeds_curves.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss_mean", "train_loss_sd", "val_loss_mean",
                    "val_loss_sd", "val_contain_mean", "val_contain_sd", "n_seeds"])
        for ep in sorted(curves):
            a = np.array(curves[ep], dtype=float)
            with np.errstate(invalid="ignore"):
                cmean = np.nanmean(a[:, 2]) if not np.all(np.isnan(a[:, 2])) else np.nan
                csd = np.nanstd(a[:, 2], ddof=1) if not np.all(np.isnan(a[:, 2])) else np.nan
            w.writerow([ep, round(a[:, 0].mean(), 6), round(a[:, 0].std(ddof=1), 6),
                        round(a[:, 1].mean(), 6), round(a[:, 1].std(ddof=1), 6),
                        "" if cmean != cmean else round(cmean, 6),
                        "" if csd != csd else round(csd, 6), len(a)])

    fields = list(rows[0].keys())
    with open(os.path.join(HERE, "results", "detector_seeds_summary.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [f"DETECTOR FINE-TUNE, {len(seeds)} SEEDS (split seed {SPLIT_SEED})",
             f"checkpoint = lowest validation loss, val loss on {args.val_loss_frames} "
             f"unaugmented validation frames", ""]
    for k in ("test_contain", "val_contain", "light_mean_contain", "best_epoch",
              "train_seconds", "collapse_hot_anchors"):
        v = np.array([r[k] for r in rows], dtype=float)
        lines.append(f"  {k:<20} mean {v.mean():>10.4f}  SD {v.std(ddof=1):>8.4f}"
                     f"  min {v.min():>8.4f}  max {v.max():>8.4f}")
    n_trip = sum(1 for r in rows if r["collapse_tripped"])
    lines.append(f"\n  collapse guard tripped for {n_trip} of {len(rows)} seeds")
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "results", "detector_seeds_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
