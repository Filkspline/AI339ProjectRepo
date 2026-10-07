"""Part 1 investigation: is "has it moved?" a usable hand-vs-clutter signal?

The idea from live testing: every remaining false lock (poster, room corner) is
on a STATIC object, and the camera is assumed stationary, so an object that has
never moved cannot be a hand -- even a hand being "held still" has micro-tremor.

ASSUMPTION (limitation, and the main caveat of this whole idea): the camera must
not move.  If the camera is bumped, panned, or the laptop is shifted, everything
in the frame moves at once and this signal calls the whole room a hand.  Nothing
here detects camera motion, and nothing downstream compensates for it.

This script measures, offline and before anything is wired into the pipeline:

  A. the clips WITH a hand: motion inside the region a detection would be judged
     on (centred on the MediaPipe hand, sized like the detector's box);
  B. the clips with NO hand: motion in EVERY tile of a grid, so a static room
     must be quiet everywhere -- including wherever the poster/corner sit;
  C. the crux case -- CloseLightStill, a genuinely present hand held still --
     against NohandLight/NohandDark, truly static rooms.  If a still hand and a
     static poster score the same, the idea does not work as described and that
     is the finding, not something to tune away;
  D. (--ae-sim) the auto-exposure / auto-white-balance failure mode: a global
     brightness change looks like motion EVERYWHERE to naive frame differencing.
     A simulated webcam gain ramp AND gain step are injected into a no-hand clip
     to show which metrics survive.

Metrics (8-bit luma, ring of RECENT frames, fixed frame coordinates so a moving
object registers as change):
    mad      mean |newest - oldest|                        (naive differencing)
    mad_bc   same after subtracting each frame's own mean  (bias compensation)
    mad_gc   old frame rescaled to the new frame's GLOBAL mean/std first, so a
             frame-wide brightness/gain change is divided out BEFORE
             differencing (this is the auto-exposure fix; the gain is estimated
             over the whole frame, which has plenty of signal, not over the
             small region, which does not)
    mad_aff  each frame affine-normalised using its OWN regional statistics
             (shown here to backfire: it amplifies sensor noise in flat/dark
             regions)
    frac     fraction of pixels with |diff| > THR, bias-compensated
    frac_gc  same, but with the global gain compensation above
    tstd     mean temporal std over the ring, bias-compensated
    mog2     MOG2 foreground fraction inside the region
    gshift   |frame mean luma change| (global lighting indicator)

Usage:
    .venv-blazepalm\\Scripts\\python.exe motion_probe.py --ae-sim
    ... --scale 0.333        (match the live 640x480 webcam)
"""
import argparse
import os
import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_HAND = ["CloseLightMoving.mp4", "CloseLightStill.mp4", "CloseDarkStill.mp4",
              "FarLight.mp4", "FarDark.mp4"]
CLIPS_NONE = ["NohandLight.mp4", "NohandDark.mp4"]
THR = 12.0          # per-pixel luma change counted as "different"
RING = 5            # frames in the ring (~170 ms at 30 fps)
GRID = 6            # grid divisions per axis for the no-hand sweep
MOG_WARMUP = 30     # frames before MOG2 is trusted
BOX_SCALE = 2.6     # the detector's decoded box is 2.6x the palm width
MIN_REGION_FRAC = 0.08
KEYS = ["mad", "mad_bc", "mad_gc", "frac", "frac_gc", "tstd", "mog2"]


@dataclass
class F:
    """One frame: luma plus its GLOBAL statistics (used for gain correction)."""
    luma: np.ndarray
    mean: float
    std: float


def make_frame(bgr):
    lu = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return F(lu, float(lu.mean()), float(lu.std()))


def crop(img, cx, cy, side):
    h, w = img.shape[:2]
    side = int(max(8, min(int(round(side)), min(h, w))))
    x0 = max(0, min(int(round(cx - side / 2.0)), w - side))
    y0 = max(0, min(int(round(cy - side / 2.0)), h - side))
    return img[y0:y0 + side, x0:x0 + side]


def region_metrics(frames, cx, cy, side, mog_mask=None, extra=None):
    """frames: list[F] oldest first; the region is a square at (cx, cy)."""
    cur = crop(frames[-1].luma, cx, cy, side)
    old = crop(frames[0].luma, cx, cy, side)
    n = float(cur.size)

    a = cur - cur.mean()
    b = old - old.mean()

    # global gain+bias alignment of the old frame onto the new one
    g = frames[-1].std / (frames[0].std + 1e-6)
    off = frames[-1].mean - g * frames[0].mean
    old_gc = g * old + off
    agc = cur - cur.mean()
    bgc = old_gc - old_gc.mean()

    an = a / (a.std() + 1e-6)
    bn = b / (b.std() + 1e-6)

    out = dict(
        mad=float(np.abs(cur - old).sum() / n),
        mad_bc=float(np.abs(a - b).sum() / n),
        mad_gc=float(np.abs(agc - bgc).sum() / n),
        mad_aff=float(np.abs(an - bn).sum() / n),
        frac=float((np.abs(a - b) > THR).sum() / n),
        frac_gc=float((np.abs(agc - bgc) > THR).sum() / n),
    )
    if mog_mask is not None:
        out["mog2"] = float(mog_mask.sum() / n)
    if extra:
        out.update(extra)
    return out


def tstd_of(frames, cx, cy, side):
    st = np.stack([crop(f.luma, cx, cy, side) - f.mean for f in frames])
    return float(st.std(axis=0).mean())


def agg(rows, key):
    v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
    if not len(v):
        return (float("nan"),) * 3
    return float(np.median(v)), float(np.percentile(v, 90)), float(v.max())


def line3(t):
    return f"{t[0]:7.3f} {t[1]:7.3f} {t[2]:7.3f}".rjust(22)


# ---------------------------------------------------------------- A: hand clips
def hand_region_lookup(gt, max_frames):
    present = gt[gt[:, 1] == 1][:, 0] if len(gt) else np.array([])
    out = {}
    for i in range(1, max_frames + 1):
        row = gt[gt[:, 0] == i]
        if len(row) and row[0, 1]:
            out[i] = (float(row[0, 2]), float(row[0, 3]), float(row[0, 4]))
            continue
        if len(present):
            near = present[np.abs(present - i) <= 5]
            if len(near):
                k = near[np.argmin(np.abs(near - i))]
                row = gt[gt[:, 0] == k]
                out[i] = (float(row[0, 2]), float(row[0, 3]), float(row[0, 4]))
    return out


def read_clip(clip, max_frames, scale, gain=None, noise=0.0, rng=None):
    """Yield (frame_index, F, bgr_or_None).  gain(n) -> multiplier for AE sims.

    `noise` adds independent per-frame Gaussian noise of that sigma, which is
    the pessimistic model of webcam sensor noise (temporally white, so it lands
    entirely in the frame difference).
    """
    cap = cv2.VideoCapture(os.path.join(HERE, clip))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret or n >= max_frames:
            break
        n += 1
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        if gain is not None:
            g = gain(n)
            if g != 1.0:
                frame = np.clip(frame.astype(np.float32) * g, 0, 255).astype(np.uint8)
        if noise:
            r = rng or np.random.default_rng(0)
            frame = np.clip(frame.astype(np.float32)
                            + r.normal(0.0, noise, frame.shape), 0, 255
                            ).astype(np.uint8)
        yield n, make_frame(frame), frame
    cap.release()


def probe_hand(clip, gt, max_frames, scale, noise=0.0):
    lookup = hand_region_lookup(gt, max_frames)
    ring = deque(maxlen=RING)
    rows = []
    rng = np.random.default_rng(1234)
    mog = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=16,
                                             detectShadows=False)
    for n, f, _ in read_clip(clip, max_frames, scale, noise=noise, rng=rng):
        ring.append(f)
        mask = mog.apply(f.luma.astype(np.uint8))
        r = lookup.get(n)
        if r is None or len(ring) < RING or n <= MOG_WARMUP:
            continue
        fw = f.luma.shape[1]
        cx, cy, pw = r[0] * scale, r[1] * scale, r[2] * scale
        side = max(BOX_SCALE * pw, MIN_REGION_FRAC * fw)
        h, w = f.luma.shape[:2]
        y0, x0 = int(max(0, cy - side / 2)), int(max(0, cx - side / 2))
        y1, x1 = int(min(h, y0 + side)), int(min(w, x0 + side))
        m = region_metrics(list(ring), cx, cy, side,
                           mog_mask=mask[y0:y1, x0:x1], extra=dict(
                               side=side / fw, tstd=tstd_of(list(ring), cx, cy, side),
                               gshift=float(abs(ring[-1].mean - ring[0].mean))))
        rows.append(m)
    return rows


# ---------------------------------------------------------------- B: no-hand clips
def probe_none(clip, max_frames, scale, noise=0.0):
    mog = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=16,
                                             detectShadows=False)
    ring = deque(maxlen=RING)
    per_tile = {}
    gshifts = []
    rng = np.random.default_rng(1234)
    for n, f, _ in read_clip(clip, max_frames, scale, noise=noise, rng=rng):
        h, w = f.luma.shape[:2]
        ring.append(f)
        mask = mog.apply(f.luma.astype(np.uint8))
        if len(ring) < RING or n <= MOG_WARMUP:
            continue
        gshifts.append(float(abs(ring[-1].mean - ring[0].mean)))
        for gy in range(GRID):
            for gx in range(GRID):
                y0, x0 = int(gy * h / GRID), int(gx * w / GRID)
                y1, x1 = int((gy + 1) * h / GRID), int((gx + 1) * w / GRID)
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                side = min(x1 - x0, y1 - y0)
                per_tile.setdefault((gy, gx), []).append(
                    region_metrics(list(ring), cx, cy, side,
                                   mask[y0:y1, x0:x1],
                                   extra=dict(tstd=tstd_of(list(ring), cx, cy,
                                                           side))))
    return per_tile, gshifts


# ---------------------------------------------------------------- D: AE sim
def probe_ae(clip, max_frames, scale, amp, mode):
    def gain(n):
        if not amp:
            return 1.0
        if mode == "sin":
            return 1.0 + amp * np.sin(2 * np.pi * n / 30.0)
        return 1.0 + amp if (n // 20) % 2 else 1.0 - amp

    ring = deque(maxlen=RING)
    rows = []
    mog = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=16,
                                             detectShadows=False)
    for n, f, _ in read_clip(clip, max_frames, scale, gain):
        h, w = f.luma.shape[:2]
        ring.append(f)
        mask = mog.apply(f.luma.astype(np.uint8))
        if n <= MOG_WARMUP or len(ring) < RING:
            continue
        cx, cy, side = w / 2.0, h / 2.0, 0.20 * w
        y0, x0 = int(cy - side / 2), int(cx - side / 2)
        y1, x1 = y0 + int(side), x0 + int(side)
        rows.append(region_metrics(list(ring), cx, cy, side,
                                   mog_mask=mask[y0:y1, x0:x1], extra=dict(
                                       tstd=tstd_of(list(ring), cx, cy, side),
                                       gshift=float(abs(ring[-1].mean - ring[0].mean)))))
    return rows


def med(rows, key):
    v = [r[key] for r in rows if key in r]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize frames before analysis, e.g. 0.333 to match "
                         "the live 640x480 webcam")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--ae-sim", action="store_true")
    ap.add_argument("--ae-amp", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="add per-frame Gaussian sensor noise of this sigma")
    args = ap.parse_args()
    scale = args.scale
    z = np.load(os.path.join(HERE, "gt_clips.npz"))

    print(f"scale {scale} | ring {RING} frames (~170 ms @30fps) | "
          f"pixel threshold {THR}/255 | up to {args.frames} frames/clip")
    print(f"hand region = box of side max({BOX_SCALE} x palm width, "
          f"{MIN_REGION_FRAC} x frame) centred on the MediaPipe hand")
    print(f"no-hand clips = every tile of a {GRID}x{GRID} grid "
          f"(MOG2 warm-up {MOG_WARMUP} frames skipped)")
    if args.noise:
        print(f"sensor noise: Gaussian sigma {args.noise}/255 per frame, "
              f"temporally white (worst case for differencing)")
    print()

    print("== A. clips WITH a hand: motion in the region a detection is judged on ==")
    print(f"{'clip':<21}{'reg/frame':>10}{'frames':>7}"
          + "".join(f"{k:>22}" for k in KEYS))
    print(f"{'':<21}{'':>10}{'':>7}"
          + "".join(f"{'med':>7}{'p90':>8}{'max':>7}" for _ in KEYS))
    print("-" * (38 + 22 * len(KEYS)))
    hand_rows = {}
    for clip in CLIPS_HAND:
        if clip not in z.files:
            continue
        rows = probe_hand(clip, z[clip], args.frames, scale, args.noise)
        hand_rows[clip] = rows
        if not rows:
            print(f"{clip:<21}  (no frames with a hand)")
            continue
        line = (f"{clip:<21}{np.median([r['side'] for r in rows]):>10.3f}"
                f"{len(rows):>7}")
        for k in KEYS:
            line += line3(agg(rows, k))
        print(line)

    print("\n== B. clips with NO hand: a static room must be quiet in EVERY tile ==")
    print("   (per-tile median over time, summarised across all tiles)")
    print(f"{'clip':<21}{'tiles':>7}"
          + "".join(f"{k:>22}" for k in KEYS))
    print(f"{'':<21}{'':>7}"
          + "".join(f"{'med':>7}{'p90':>8}{'max':>7}" for _ in KEYS))
    print("-" * (28 + 22 * len(KEYS)))
    none_stats = {}
    for clip in CLIPS_NONE:
        per_tile, gshifts = probe_none(clip, args.frames, scale, args.noise)
        if not per_tile:
            print(f"{clip:<21}  (no frames)")
            continue
        line = f"{clip:<21}{len(per_tile):>7}"
        for k in KEYS:
            v = np.array([np.median([m[k] for m in ms]) for ms in per_tile.values()])
            none_stats.setdefault(k, {})[clip] = (float(np.median(v)),
                                                  float(np.percentile(v, 90)),
                                                  float(v.max()))
            line += line3((float(np.median(v)), float(np.percentile(v, 90)),
                           float(v.max())))
        print(line)

    print("\n== C. CRUX: a present-but-STILL hand vs a truly STATIC room ==")
    print("   CloseLightStill/CloseDarkStill are real hands held still.")
    print(f"   {'region':<38}{'mad_bc':>9}{'mad_gc':>9}{'frac':>8}{'frac_gc':>9}{'tstd':>8}")
    for clip in CLIPS_HAND:
        rows = hand_rows.get(clip)
        if not rows:
            continue
        tag = ("HAND moving " if "Moving" in clip
               else "HAND far " if clip.startswith("Far") else "HAND held-still ")
        print(f"   {tag + clip:<38}"
              f"{np.median([r['mad_bc'] for r in rows]):>9.3f}"
              f"{np.median([r['mad_gc'] for r in rows]):>9.3f}"
              f"{np.median([r['frac'] for r in rows]):>8.3f}"
              f"{np.median([r['frac_gc'] for r in rows]):>9.3f}"
              f"{np.median([r['tstd'] for r in rows]):>8.3f}")
    for clip in CLIPS_NONE:
        if clip not in none_stats["frac"]:
            continue
        print(f"   {'STATIC ROOM ' + clip:<38}"
              f"{none_stats['mad_bc'][clip][0]:>9.3f}"
              f"{none_stats['mad_gc'][clip][0]:>9.3f}"
              f"{none_stats['frac'][clip][0]:>8.3f}"
              f"{none_stats['frac_gc'][clip][0]:>9.3f}"
              f"{none_stats['tstd'][clip][0]:>8.3f}")
        print(f"   {'   (worst tile of that room)':<38}"
              f"{none_stats['mad_bc'][clip][2]:>9.3f}"
              f"{none_stats['mad_gc'][clip][2]:>9.3f}"
              f"{none_stats['frac'][clip][2]:>8.3f}"
              f"{none_stats['frac_gc'][clip][2]:>9.3f}"
              f"{none_stats['tstd'][clip][2]:>8.3f}")

    print("\n   separation: still-hand median / worst static-room tile")
    print(f"   {'hand clip':<24}" + "".join(f"{k:>12}" for k in
                                            ("mad_bc", "frac", "frac_gc", "tstd")))
    for clip in ("CloseLightStill.mp4", "CloseDarkStill.mp4", "FarDark.mp4"):
        rows = hand_rows.get(clip)
        if not rows:
            continue
        line = f"   {clip:<24}"
        for k in ("mad_bc", "frac", "frac_gc", "tstd"):
            h = float(np.median([r[k] for r in rows]))
            worst = max(none_stats[k][c][2] for c in CLIPS_NONE
                        if c in none_stats.get(k, {}))
            line += f"{h / worst if worst else float('inf'):>12.2f}"
        print(line)

    # ---- raw precision: how quiet is a static room really? ---------------
    print("\n   exact worst static-room tile values (not rounded):")
    for clip in CLIPS_NONE:
        if clip not in none_stats["frac"]:
            continue
        print(f"     {clip:<22} frac_gc median {none_stats['frac_gc'][clip][0]:.6f}"
              f"  p90 {none_stats['frac_gc'][clip][1]:.6f}"
              f"  worst {none_stats['frac_gc'][clip][2]:.6f}")

    # ---- gate sweep ------------------------------------------------------
    print("\n== E. gate sweep: what a threshold on frac_gc would do ==")
    print("   hand clips: % of frames WITH a hand that pass")
    print("   no-hand clips: % of room tiles that would FALSELY pass\n")
    grid = [r for ms in [hand_rows[c] for c in CLIPS_HAND if hand_rows.get(c)]
            for r in ms]
    tiles = []
    for clip in CLIPS_NONE:
        if clip not in none_stats["frac_gc"]:
            continue
        # rebuild per-tile series for the sweep (cheap: reuse stored medians)
        tiles.append((clip, none_stats["frac_gc"][clip]))
    print(f"   {'thr':>7}" + "".join(f"{c.split('.')[0][:11]:>13}"
                                     for c in CLIPS_HAND) + f"{'tiles>thr':>11}")
    for thr in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
        line = f"   {thr:>7.3f}"
        for clip in CLIPS_HAND:
            rows = hand_rows.get(clip)
            if not rows:
                line += f"{'--':>13}"
                continue
            line += f"{100*np.mean([r['frac_gc'] >= thr for r in rows]):>12.0f}%"
        over = sum(1 for _c, st in tiles if st[2] >= thr)
        line += f"{over:>11}"
        print(line)
    print("\n   ('tiles>thr' counts rooms' WORST tile passing, which is the "
          "pessimistic 0-or-2 case)")

    if args.ae_sim:
        print(f"\n== D. auto-exposure / auto-white-balance failure mode "
              f"(+-{args.ae_amp:.0%} global gain) ==")
        print("   a NO-HAND clip, so every movement here is a false positive")
        for mode in ("sin", "step"):
            print(f"\n   -- {mode} gain --")
            for clip in CLIPS_NONE:
                clean = probe_ae(clip, args.frames, scale, 0.0, mode)
                dirty = probe_ae(clip, args.frames, scale, args.ae_amp, mode)
                print(f"   {clip}")
                print(f"      {'metric':<10}{'clean':>9}{'with AE':>10}{'ratio':>8}")
                for k in KEYS + ["mad_aff"]:
                    a, b = med(clean, k), med(dirty, k)
                    r = b / a if a else float("nan")
                    print(f"      {k:<10}{a:>9.3f}{b:>10.3f}{r:>8.2f}")
                print(f"      {'gshift':<10}{med(clean, 'gshift'):>9.2f}"
                      f"{med(dirty, 'gshift'):>10.2f}"
                      f"   <- global lighting indicator")


if __name__ == "__main__":
    sys.exit(main())
