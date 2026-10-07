"""Step 2: fine-tune the palm detector for dark / far conditions (DESIGN IN CODE).

DESIGN
------
What is unfrozen: by default ONLY the six 1x1 prediction heads
(`class_32/16/8`, `reg_32/16/8`).  Rationale: the labelled dark data is ~200 real
frames from one clip, so the largest safe parameter set is the smallest one.  The
heads are also where a domain shift can be absorbed as a per-anchor offset/scale
without touching the features the rest of the pipeline depends on.
`--unfreeze heads_last` additionally unfreezes `scaled32add`, the last residual
block, which runs on the finest (32x32) feature map.  Spatial detail has already
collapsed by the 8x8 stage, so nothing deeper is worth unfreezing -- and the
default keeps the general-purpose detector as intact as possible.  Compare the two
with the built-in evaluation rather than assuming.

What is frozen: backbone1/2/3, both scale paths, `scaled16add`.

Losses: BCE over all 2944 anchors (pos_weight against the ~1:3000 imbalance) plus
smooth-L1 on the BOX and on KEYPOINTS 0 (wrist) and 2 (middle MCP) for anchors
whose centre falls inside the ground-truth box.  Only those two keypoints are
supervised: they are the only ones the port's ROI code consumes and the only ones
MediaPipe's landmarks map onto unambiguously (0 and 9).  The other ten keypoint
outputs keep their pretrained values rather than being taught a guess.  Targets use
the port's own encoding, read out of `_decode_boxes`, so the decoder stays valid.

Data: VeryLowLight frames with lag-free MediaPipe labels, TRAIN BLOCKS ONLY, with
the established dark/distance augmentation (photometric match to the clip's
measured statistics x three severities x three distance bands, plus mild
brightness/blur jitter).  All of it is geometry-preserving, so labels stay correct.

Held out, never trained on: the VeryLowLight holdout blocks (every 3rd block of 50
frames) and all seven original clips -- which is what makes the light-condition
trade honest rather than in-sample.

GENERALIZATION RISK, stated up front: the real dark data is one room, one lamp,
one camera, ~200 frames.  Any gain on the VeryLowLight holdout is cross-frame, not
cross-condition.  The script therefore prints the train-block vs holdout gap
(memorisation shows up as a number) and the same metric on the untouched light
clips, and refuses to be optimistic about either.

Usage:
    .venv-blazepalm\\Scripts\\python.exe finetune_detector_dark.py --unfreeze heads
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from blazepalm import PalmDetector                          # noqa: E402
from hand_mouse_cursor import letterbox                     # noqa: E402
from build_dark_dataset import make_variant                 # noqa: E402
from build_vll_dataset import target_stats                  # noqa: E402

BLOCK, HOLDOUT_EVERY = 50, 3
LIGHT_CLIPS = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
               "FarLight.mp4", "FarDark.mp4", "NohandLight.mp4", "NohandDark.mp4"]


def is_holdout(n):
    return (n // BLOCK) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1


def to_tensor(bgr256):
    """Exactly the port's preprocessing: BGR kept as-is, CHW, /127.5 - 1."""
    t = torch.from_numpy(np.ascontiguousarray(bgr256)).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


def _np(a):
    """Accept either the tensor stored on the detector or a plain array."""
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def gt_targets(box_px, kps, anchors, scale, left, top):
    """Encode a GT box + 2 keypoints into the port's raw-output space.

    Inverse of PalmDetector._decode_boxes (anchors are normalised, w = h = 1.0):
        x_c = raw0/256*aw + ax       y_c = raw1/256*ah + ay - h/5.2
        w   = raw2/256*aw*2.6        h   = raw3/256*ah*2.6
        kp  = raw/256*a + a_offset
    """
    anchors = _np(anchors)
    A = anchors.shape[0]
    x0, y0, x1, y1 = box_px
    xc = ((x0 + x1) / 2.0 * scale + left) / 256.0
    yc = ((y0 + y1) / 2.0 * scale + top) / 256.0
    w = (x1 - x0) * scale / 256.0
    h = (y1 - y0) * scale / 256.0
    t = np.zeros((A, 18), dtype=np.float32)
    m = np.zeros((A, 18), dtype=np.float32)
    t[:, 0] = (xc - anchors[:, 0]) * 256.0
    t[:, 1] = (yc + h / 5.2 - anchors[:, 1]) * 256.0
    t[:, 2] = w / 2.6 * 256.0          # anchor w == 1.0 for every anchor
    t[:, 3] = h / 2.6 * 256.0
    m[:, 0:4] = 1.0
    for j, (kx, ky) in ((0, kps[0]), (2, kps[1])):
        t[:, 4 + 2 * j] = ((kx * scale + left) / 256.0 - anchors[:, 0]) * 256.0
        t[:, 5 + 2 * j] = ((ky * scale + top) / 256.0 - anchors[:, 1]) * 256.0
        m[:, 4 + 2 * j] = 1.0
        m[:, 5 + 2 * j] = 1.0
    return t, m


def match_anchors(box_px, scale, left, top, anchors, max_anchors=6):
    """Label only the few anchors nearest the ground-truth box centre.

    The naive rule "every anchor whose centre is inside the box" labels ~96 of
    2944 anchors positive for a large 2.6x-palm box, and with a heavy pos_weight
    that diffuse signal teaches the score head to fire everywhere -- measured:
    the first attempt raised anchors >= 0.5 from 13 to 337 on a light frame, i.e.
    it destroyed precision and made the top detection useless.  The port's anchors
    are all unit-size, so MediaPipe's scale-aware matching is not available; the
    available lever is to keep the positive set small and central.
    """
    anchors = _np(anchors)
    x0, y0, x1, y1 = box_px
    bcx = ((x0 + x1) / 2.0 * scale + left) / 256.0
    bcy = ((y0 + y1) / 2.0 * scale + top) / 256.0
    d = ((anchors[:, 0] - bcx) ** 2 + (anchors[:, 1] - bcy) ** 2)
    k = int(min(max_anchors, len(d)))
    idx = np.argsort(d)[:k]
    # only keep anchors that are actually inside the box (with a little slack)
    inside = np.array([(x0 - 0.05 * (x1 - x0)) * scale + left <= anchors[i, 0] * 256
                       <= (x1 + 0.05 * (x1 - x0)) * scale + left
                       and (y0 - 0.05 * (y1 - y0)) * scale + top
                       <= anchors[i, 1] * 256
                       <= (y1 + 0.05 * (y1 - y0)) * scale + top for i in idx])
    return idx[inside]


def anchor_centre_in_box(box_px, scale, left, top, anchors):
    anchors = _np(anchors)
    x0, y0, x1, y1 = box_px
    ax = anchors[:, 0] * 256.0
    ay = anchors[:, 1] * 256.0
    return ((ax >= x0 * scale + left) & (ax <= x1 * scale + left)
            & (ay >= y0 * scale + top) & (ay <= y1 * scale + top))


def eval_clip(det, clip, gt, frames=100000, dark_aug=False, rng=None,
              t_mean=None, t_std=None, holdout_only=False):
    """Detector-only localisation: does the TOP detection contain the hand?"""
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    n = hit = fired = handed = 0
    while True:
        ret, frame = cap.read()
        if not ret or n >= frames:
            break
        n += 1
        if holdout_only and not is_holdout(n):
            continue
        row = gt[gt[:, 0] == n]
        if not len(row) or not row[0, 1]:
            continue
        handed += 1
        img = frame
        if dark_aug and t_mean is not None:
            img, _s, _b = make_variant(frame, t_mean, t_std, rng)
        padded, scale, left, top = letterbox(img, 256)
        with torch.no_grad():
            c, r = det(to_tensor(padded).unsqueeze(0))
        boxes = det._decode_boxes(r[0], det.anchors)
        scores = c[0, :, 0].sigmoid()
        k = int(torch.argmax(scores))
        if float(scores[k]) < 0.5:
            continue
        fired += 1
        b = boxes[k]
        # _decode_boxes returns NORMALISED coords of the 256x256 letterboxed
        # input (anchors are normalised), so multiply by 256 before undoing the
        # letterbox -- the same thing hand_pipeline.box_to_pixels does.
        bx0 = (float(b[0]) * 256.0 - left) / scale
        by0 = (float(b[1]) * 256.0 - top) / scale
        bx1 = (float(b[2]) * 256.0 - left) / scale
        by1 = (float(b[3]) * 256.0 - top) / scale
        cx, cy = float(row[0, 2]), float(row[0, 3])
        w, h = bx1 - bx0, by1 - by0
        if (bx0 - 0.15 * w <= cx <= bx1 + 0.15 * w
                and by0 - 0.15 * h <= cy <= by1 + 0.15 * h):
            hit += 1
    cap.release()
    return dict(handed=handed, fired=fired, hit=hit,
                contain=hit / max(handed, 1), fire_rate=fired / max(handed, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ML, "palmdetector.pth"))
    ap.add_argument("--out", default="palmdetector_dark.pth")
    ap.add_argument("--clip", default="VeryLowLight.mp4")
    ap.add_argument("--gt", default="gt_verylowlight.npz")
    ap.add_argument("--gt-key", default="static_conf0.3")
    ap.add_argument("--lm-key", default="static_conf0.3_landmarks")
    ap.add_argument("--unfreeze", choices=["heads", "heads_last"], default="heads")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=5.0,
                    help="kept modest: the positive set is now a few "
                         "central anchors, not ~96 box-covering ones")
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--light-frac", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t_mean, t_std, _ = target_stats(args.clip)

    det = PalmDetector()
    det.load_weights(args.weights)
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    for p in det.parameters():
        p.requires_grad = False
    names = ["class_32", "class_16", "class_8", "reg_32", "reg_16", "reg_8"]
    if args.unfreeze == "heads_last":
        names.append("scaled32add")
    for nm in names:
        for p in getattr(det, nm).parameters():
            p.requires_grad = True
    n_tr = sum(p.numel() for p in det.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in det.parameters())
    print(f"unfreezing: {', '.join(names)}")
    print(f"trainable: {n_tr:,} / {n_tot:,} params ({100*n_tr/n_tot:.2f}%)")

    z = np.load(os.path.join(HERE, args.gt))
    gt = z[args.gt_key]
    lm_all = z[args.lm_key]
    cap = cv2.VideoCapture(os.path.join(HERE, args.clip))
    train, holdout = [], []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        row = gt[gt[:, 0] == n]
        if not (len(row) and row[0, 1] and not np.isnan(lm_all[n - 1]).any()):
            continue
        (holdout if is_holdout(n) else train).append((n, frame.copy()))
    cap.release()
    print(f"labelled frames: {len(train)} train-block, {len(holdout)} holdout-block")
    if len(train) < 20:
        raise SystemExit("too few labelled training frames")

    def sample(rg):
        n_, frame = train[int(rg.integers(len(train)))]
        row = gt[gt[:, 0] == n_]
        fw, fh = frame.shape[1], frame.shape[0]
        cx, cy, pw = float(row[0, 2]), float(row[0, 3]), float(row[0, 4])
        box = (cx - 1.3 * pw, cy - 1.3 * pw, cx + 1.3 * pw, cy + 1.3 * pw)
        L = lm_all[n_ - 1]
        kps = ((float(L[0, 0]) * fw, float(L[0, 1]) * fh),
               (float(L[9, 0]) * fw, float(L[9, 1]) * fh))
        img = frame
        if rg.random() > args.light_frac:
            img, _s, _b = make_variant(frame, t_mean, t_std, rg)
        return img, box, kps

    params = [p for p in det.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    steps = max(1, len(train) // args.batch)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        det.train()
        run_cls = run_reg = 0.0
        for _ in range(steps):
            cb, rb, mb, tb = [], [], [], []
            for _b in range(args.batch):
                img, box, kps = sample(rng)
                padded, scale, left, top = letterbox(img, 256)
                t, m = gt_targets(box, kps, det.anchors, scale, left, top)
                idx = match_anchors(box, scale, left, top, det.anchors)
                if len(idx) == 0:
                    continue
                tgt_pos = np.zeros(det.anchors.shape[0], dtype=np.float32)
                tgt_pos[idx] = 1.0
                cb.append(padded)
                rb.append(t)
                mb.append(m)
                tb.append(tgt_pos)
            if not cb:
                continue
            x = torch.stack([to_tensor(p) for p in cb])
            c, r = det(x)
            tgt = torch.from_numpy(np.stack(tb))
            cls = bce(c[:, :, 0], tgt) * (1.0 + (args.pos_weight - 1.0) * tgt)
            reg_m = torch.from_numpy(np.stack(mb)) * tgt.unsqueeze(-1)
            diff = F.smooth_l1_loss(r, torch.from_numpy(np.stack(rb)),
                                    reduction="none") * reg_m
            loss = cls.mean() + args.reg_weight * (diff.sum()
                                                  / reg_m.sum().clamp(min=1.0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_cls += float(cls.mean())
            run_reg += float(diff.sum() / reg_m.sum().clamp(min=1.0))
        msg = (f"  epoch {ep:>3}/{args.epochs} cls {run_cls/steps:.4f} "
               f"reg {run_reg/steps:.4f} ({time.time()-t0:.0f}s)")
        if args.eval_every and (ep % args.eval_every == 0 or ep == args.epochs):
            det.eval()
            e_tr = eval_clip(det, args.clip, gt, holdout_only=False,
                             dark_aug=True, rng=rng, t_mean=t_mean, t_std=t_std)
            e_ho = eval_clip(det, args.clip, gt, holdout_only=True,
                             dark_aug=True, rng=rng, t_mean=t_mean, t_std=t_std)
            msg += (f" | train-contain {100*e_tr['contain']:.1f}% "
                    f"holdout-contain {100*e_ho['contain']:.1f}%")
        print(msg, flush=True)
    # ---- collapse guard --------------------------------------------------
    # The first attempt raised anchors >= 0.5 from 13 to 337 on a light frame and
    # was silently useless.  Check sparsity on a light frame before saving.
    probe = cv2.VideoCapture(os.path.join(HERE, "CloseLightMoving.mp4"))
    ok, pf = probe.read()
    probe.release()
    n_hot = -1
    if ok:
        pp, _sc, _l, _t = letterbox(pf, 256)
        with torch.no_grad():
            cc, _rr = det(to_tensor(pp).unsqueeze(0))
        hot = cc[0, :, 0].sigmoid()
        n_hot = int((hot >= 0.5).sum())
        print(f"\nsparsity guard: {n_hot} anchors >= 0.5 on a light frame "
              f"(original is ~13; a collapsed model fires on hundreds)")
        if n_hot > 100:
            os.rename(os.path.join(HERE, args.out),
                      os.path.join(HERE, args.out + ".collapsed"))
            print("   REJECTED: saved as *.collapsed instead of " + args.out)
            print("   the fine-tune collapsed the score head; do not ship it")
            return
    torch.save(det.state_dict(), os.path.join(HERE, args.out))
    print(f"\nsaved -> {args.out}")

    # ---- the trade, measured: original vs fine-tuned, dark and light -----
    orig = PalmDetector()
    orig.load_weights(args.weights)
    orig.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    print("\n== detector-only localisation: box-contains-hand ==")
    print(f"   {'clip':<24}{'frames':>8}{'orig':>9}{'fine-tuned':>12}{'delta':>9}")
    rows = [(args.clip, gt, True), (args.clip, gt, False)]
    for clip, g_, _h in rows:
        pass
    out = []
    e_dark_o = eval_clip(orig, args.clip, gt, dark_aug=True, rng=rng,
                         t_mean=t_mean, t_std=t_std)
    e_dark_n = eval_clip(det, args.clip, gt, dark_aug=True, rng=rng,
                         t_mean=t_mean, t_std=t_std)
    print(f"   {'VeryLowLight (aug)':<24}{e_dark_o['handed']:>8}"
          f"{100*e_dark_o['contain']:>8.1f}%{100*e_dark_n['contain']:>11.1f}%"
          f"{100*(e_dark_n['contain']-e_dark_o['contain']):>+8.1f}")
    out.append(("VeryLowLight (aug)", e_dark_o, e_dark_n))
    for clip in LIGHT_CLIPS:
        g_ = np.load(os.path.join(HERE, "gt_clips.npz"))[f"static:{clip}"]
        eo = eval_clip(orig, clip, g_)
        en = eval_clip(det, clip, g_)
        print(f"   {clip:<24}{eo['handed']:>8}{100*eo['contain']:>8.1f}%"
              f"{100*en['contain']:>11.1f}%"
              f"{100*(en['contain']-eo['contain']):>+8.1f}")
        out.append((clip, eo, en))
    d = [n["contain"] - o["contain"] for _c, o, n in out]
    print(f"\n   light-clip mean delta: "
          f"{100*np.mean(d[1:]):+.1f} points   dark-clip delta: {100*d[0]:+.1f} points")


if __name__ == "__main__":
    main()
