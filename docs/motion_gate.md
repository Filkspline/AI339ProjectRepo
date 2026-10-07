# Motion gate design (IMPLEMENTED in v5)

A "has this thing ever moved?" test, used as an **acquisition** gate. The idea
came from live testing: every remaining false lock (poster, room corner) is on a
static object, and even a hand that is being held still has micro-tremor.

Part 1 measured whether that is true on our footage (`motion_probe.py`). It is,
with two caveats that shaped this design. The gate is implemented in
`hand_mouse_cursor.py` (`MotionGate`, wired into `HandLocator`) and shipped as
`HandCursor_v5`; `--no-motion-gate` is the rollback. Integrated results are in
the last section.

## ASSUMPTION / LIMITATION (read this first)

**This whole approach assumes the camera does not move.** If the camera is
bumped, panned, tilted, or the laptop is shifted, then *everything* in the frame
moves at once and the motion signal reports "hand" about the entire room. Nothing
in this design detects camera motion, and no downstream stage compensates for it.
The correct behaviour on suspected camera motion is to **stop acquiring** (see
"global-change detector" below), not to trust the signal.

Two secondary failure modes, both measured:

* **Auto-exposure / auto-white-balance.** A global brightness change is motion
  "everywhere at once" to naive frame differencing. Measured on a no-hand clip
  with a +-15% gain ramp: naive mean-abs-difference inflates **7.9x**
  (0.97 -> 7.66). The metric chosen below stays at **0.000**, and a
  frame-wide gain correction takes the mean-abs-difference variant from 2.81x
  down to 1.21x. So this failure mode is survivable *if* bias compensation is
  used and the metric is the thresholded fraction, not the mean difference.
* **Sensor noise.** The metric's whole margin is a noise margin. Measured on the
  static-room clips: the changed-pixel fraction is 0.000 at sensor sigma 0 and
  0.0044 at sigma 4/255, then collapses to 0.117 at sigma 8 (where a still hand
  scores ~0.15, i.e. no separation at all). The recorded footage sits in the
  safe band, but this must be re-checked on the actual camera/room.

## Part 1 evidence (what the design is built on)

`frac` = fraction of pixels in the region whose bias-compensated luma changed by
more than 12/255 between the oldest and newest frame of a 5-frame ring, computed
on the **640x480** frames the live pipeline actually uses:

| region | frac (median) | note |
|---|---|---|
| CloseLightMoving (hand moving) | 0.534 | |
| FarLight (hand far) | 0.381 | |
| FarDark (hand far, dark) | 0.136 | |
| CloseLightStill (hand HELD STILL) | 0.076 | the crux case |
| CloseDarkStill (hand held still, dark) | 0.036 | the crux case, weakest |
| NohandLight, every one of 36 tiles | 0.000 | static room |
| NohandDark, every one of 36 tiles | 0.000 | static room |

Metrics that FAILED the crux test and are therefore not used:

| metric | still hand / worst static tile | verdict |
|---|---|---|
| mean abs diff (bias-compensated) | 0.94-1.89 | no separation |
| temporal std over the ring | 0.79-1.44 | no separation |
| per-region affine normalisation | 0.20-0.21 | inverted (amplifies noise) |
| mean abs diff, at native 1920x1280 | 0.94-1.65 | no separation |

At native sensor resolution the average-magnitude metrics invert; only the
thresholded fraction survives (2.1-5.8x). That is why the resolution matters and
why the metric is a *fraction of changed pixels*, not "how much did it change".

### The motion ring must be computed on a DOWNSCALED frame

Measured on the 640x480 clips at three analysis widths (same source footage, so
this is a controlled comparison):

| analysis width | static room, worst tile | CloseLightStill | CloseDarkStill |
|---|---|---|---|
| 640 (raw capture) | 0.016 | 0.083 | 0.036 |
| **320 (INTER_AREA)** | **0.000** | **0.078** | **0.036** |
| 160 | 0.000 | 0.082 | 0.038 |

Downscaling to ~320 wide averages sensor noise away, which drives the static
room to exactly zero, while the still hand's score is unchanged (0.083 -> 0.078
-> 0.082). The tremor signal is spatially coherent over many pixels and survives
averaging; per-pixel noise does not. So: **build the ring from a 320-wide
INTER_AREA downscale of the frame, whatever resolution the camera delivers.**
This is not an optimisation detail - at raw 640 width the weakest case
(CloseDarkStill) is only 2.25x above the noisiest static tile, and at raw
1920x1080 (which is what FarLight.mp4 was recorded at) the average-magnitude
metrics invert outright. It also cuts the motion cost 4x.

Method choice: **frame differencing**, not MOG2, not optical flow. MOG2 measured
equally well here (0.000 on every static tile, 2-42% foreground on hands) but is
a lot more machinery: it needs a full-frame model that must be applied to
every frame even though only one small region is being judged, it needs a warm-up
before it is trustworthy, and it carries state across lock changes. Dense optical
flow (Farneback) is not justified: the simple method separates all our known
cases with a >=3.4x margin at 640x480, which is the bar the brief set for even
considering flow.

## Where it sits in the pipeline

```
REACQUIRE / SEARCHING
  full-frame detect
    -> 0.15 box-size gate            (unchanged)
    -> MOTION GATE  <-- NEW, strict: the candidate's region must have moved
    -> REACQUIRE_HITS consecutive verified hits
    -> commit to TRACK

TRACK / LOCKED
  windowed detect
    -> crop verifier  P(hand) >= 0.30   (unchanged, v3)
    -> motion gate  <-- RELAXED/DROPPED: a confirmed hand may go still
    -> plausibility gate (3 frame-widths/s)
    -> heavy EMA smoothing (tau 0.35 s)
    -> speed-capped cursor (cap 1 screen-width/s, instant lock, hold-on-loss)
```

Placement relative to the existing verifier is deliberate:

* the motion test runs on the **candidate detection's own box**, the same region
  the CNN verifier crops - so both gates judge the same thing;
* it runs **after** the cheap geometric gates and **before** the CNN, so a static
  candidate is rejected without a network forward pass;
* it is **ANDed** with the CNN verifier, which is what bounds the damage on a
  false-positive motion reading: camera motion or an AE event can make the motion
  test say "moved", but the clutter still has to pass the classifier to lock.

## The motion gate itself

```
ring: last 5 frames of the 320-wide INTER_AREA downscale (~100 KB each)

on a candidate box B (in frame px, scaled to ring coordinates):
    region   = B expanded to a square of side max(2.6 x palm width, 0.08 x frame)
               (i.e. the detector's own box, with a floor so a far hand is not
                a postage stamp)
    frac     = fraction of pixels with |(cur - mean_cur) - (old - mean_old)| > 12

STRICT (acquisition):   frac >= MOTION_MIN            MOTION_MIN = 0.01
RELAXED (TRACK):        no motion requirement at all  (see below)
```

Why 0.01: at 640x480 the static room measures exactly 0.000 in every tile, while
the weakest still-hand case (CloseDarkStill, dark) measures 0.036 - a 3.6x
margin - and the common held-still case measures 0.076. A gate at 0.01 keeps
97-100% of all hand frames in the clips and admits no static tile. Raising it to
0.02-0.03 trades hand recall for nothing measurable, since static clutter is
already at 0.000.

### Why the asymmetry is safe (the re-opened-vulnerability question)

Dropping the motion requirement once TRACKING could let a static object sneak in
during a brief motion spike and then persist, because nothing re-checks. Three
things bound that, and only the first is a design choice:

1. **Acquisition is still conjunctive.** A lock needs REACQUIRE_HITS consecutive
   frames that pass the size gate AND the motion gate AND (once TRACKING starts)
   the CNN verifier. A one-frame AE/noise spike cannot satisfy a *consecutive*
   run, and after the run the object must still satisfy the verifier.
2. **A static object cannot produce `frac >= 0.01` on its own.** Something else
   must move in that region: the camera, a global lighting change, or a real
   moving object overlapping the box. The residual hole is therefore *camera
   motion and AE*, not the relaxation.
3. **The relaxation mirrors hold-on-miss**, which is already the design's
   behaviour: a confirmed hand is allowed to be momentarily undetectable without
   losing the lock. Being momentarily still is the same class of event.

If live testing shows a lock that is stale (locked, motionless, low P(hand)),
add the **liveness check** below rather than making the acquisition gate stricter.

### Proposed guards (all cheap, all optional)

* **Global-change detector.** Track `|mean luma change|` between CONSECUTIVE
  frames in the ring (not just oldest-vs-newest: an alternating AE step is
  invisible to an endpoint comparison, which is exactly how the simulated step
  measured 0.05 - no signal - while the ramp measured 11.73). If the frame-wide
  value exceeds `GLOBAL_CHANGE_MAX` (~3 luma levels), the frame is either an AE
  event or camera motion: **suspend acquisition** for that frame and (for camera
  motion) invalidate any existing lock.
* **Frame-wide gain compensation** before differencing (rescale the old frame to
  the new frame's global mean/std). Measured benefit: it is a no-op on clean
  footage and cuts the AE-ramp inflation of the mean-difference variant from
  2.81x to 1.21x. It costs two scalar ops.
* **Background-motion self-calibration.** Compute `frac` over a coarse grid of
  regions covering the frame and require the candidate region to beat the
  *median* background tile by a factor (e.g. 3x) in addition to the absolute
  floor. On a clean static room the background is 0.000 so this is a no-op; on a
  noisy camera it adapts instead of silently letting everything through.
* **Liveness check while TRACKING (only if needed).** If the tracked region has
  been motionless for `LIVENESS_TIME` (~2 s) AND the verifier's P(hand) is below
  a stricter bar (e.g. 0.6) for the same period, drop the lock. This directly
  targets "locked onto a poster forever" without weakening normal tracking.

## Cost

Ring buffer 1.5 MB, one grayscale conversion + one crop + one threshold + one
count per candidate. No extra model, no extra forward pass, no history to warm
up. Negligible next to the existing 256x256 detector pass.

## What is not in this design

* **No verifier retraining.** Part 4 is not needed: the motion signal works as a
  classical CV gate, so the CNN does not need to learn temporal information, and
  no synthetic-jitter data has to be invented (which would risk the classifier
  learning "synthetic jitter pattern" instead of "real motion"). The verifier's
  measured weakness (dark/blurry shortcut, own-room data being all close-up hands)
  is untouched by this change and stays a known limitation.
* **No optical flow**, no MOG2, no camera-motion estimation.

## Rollback

A constructor parameter (`motion_gate=None`) plus a CLI flag (`--no-motion-gate`)
restores pre-v5 behaviour, mirroring how the verifier is wired. Since v6 it also
restores the 0.15 acquisition size gate, so the flag remains a true v4 rollback
rather than a config that is *looser* than v4.

## Integrated validation (v6, `eval_motion_gate.py`, all 7 conditions, GT-matched)

The real locator (detector + verifier + motion gate) replayed over every clip --
250 frame cap, but every clip runs to its natural end, so this is the same 1000
frames the v3/v4 baseline was measured on - scored against MediaPipe ground
truth with the same hand-scaled match radius (0.12 floor + 0.6 x palm width):

| clip | v3/v4 hit / FP | **v6 hit / FP** | v6 acquisitions | motion frac at acq |
|---|---|---|---|---|
| CloseLightMoving | 159 / 1 | 156 / 1 | 1 (on-hand) | 0.632 |
| CloseLightStill | 209 / 0 | 180 / 0 | 1 (on-hand) | 0.146 |
| CloseDarkStill | 87 / 0 | 83 / 0 | 1 (on-hand) | 0.044 |
| FarLight | 25 / 26 | **28 / 6** | 1 | 0.455 |
| FarDark | 6 / 3 | **22 / 8** | 1 | 0.018 |
| NohandLight | 0 / 0 | **0 / 0** | **0** | - |
| NohandDark | 0 / 2 | **0 / 0** | **0** | - |
| **TOTAL** | 486 / 32 | **469 / 15** | 5 | |

Config comparison, same matching and same frames throughout:

| config | REACQUIRE gate | motion gate | hand frames | clutter locks |
|---|---|---|---|---|
| v3/v4 | 0.15 | off | 486 | 32 |
| v5 | 0.15 | on | 438 | 3 |
| **v6** | **0.10** | **on** | **469** | **15** |

v6 tracks 97% of the baseline's hand frames while cutting clutter locks by 53%,
and both far clips are now better than the baseline on *both* counts (FarLight
28/6 vs 25/26, FarDark 22/8 vs 6/3) - this time from an acquisition that is
genuinely on the hand rather than a clutter bootstrap. The cost is concentrated
in one place: CloseLightStill loses 29 frames, because the motion gate needs ~30
frames (~1 s) of micro-tremor before it will acquire.

**The non-regression requirement holds**: both no-hand clips produce 0
acquisitions, 0 accepted frames and 0 clutter locks at 0.10 - and did so at
every gate value down to 0.06 in the sweep.

**AE immunity at 0.10**: a +-15% gain ramp injected into the clips produced no
extra acquisition anywhere (NohandLight 0 -> 0, NohandDark 0 -> 0, FarDark 1 -> 1,
FarLight 1 -> 1). FarDark went from 1 to 4 candidates crossing the absolute floor,
contained by the gain compensation and the background calibration. NohandLight had
23 REACQUIRE frames suspended by the guard and NohandDark had 0 - expected, since
both thresholds are absolute luma levels (12/255 per pixel, 3.0 on the frame mean)
and a dark scene (mean luma 68 vs 146) moves ~2.2 instead of ~4.6 levels per frame
under the same ramp. The two thresholds are self-consistent: AE only manufactures
false motion when pixel changes exceed 12, which in practice coincides with a mean
shift above 3.0. A relative (percentage-of-mean) guard would be scale-invariant
and is a small v7 candidate, not a known failure.

Unit level, the gain compensation is decisive: on an asymmetric high-contrast
scene with a 25% gain change over the ring, the compensated changed-pixel
fraction is 0.0000 against 1.0000 without it.

## History: how v6 was reached

v5 shipped the motion gate alone, with the acquisition size gate left at 0.15.
That took FarLight from 93% TRACK to 0%: a far hand is ~0.085 of the frame, so it
can never pass a 0.15 gate; the far "tracking" of v3/v4 was bootstrapped by a lock
on static clutter big enough to pass the gate, with the hand then falling inside
the resulting search window. The motion gate correctly refused that bootstrap.

v6 relaxes the acquisition size gate to 0.10 and ties it to the motion gate in
code, which restores far-range acquisition. Measured before shipping
(`--sweep-acquire-gate`, gate ON, 200 frames):

| REACQUIRE size gate | FarLight acq / hand frames / clutter locks | NohandLight | NohandDark |
|---|---|---|---|
| 0.15 (v5) | 0 / 0 / 0 | 0 acq | 0 acq |
| **0.10 (v6)** | **1 / 27 / 7** | 0 acq | 0 acq |
| 0.08 | 1 / 27 / 7 | 0 acq | 0 acq |
| 0.06 | 1 / 26 / 6 | 0 acq | 0 acq |

## Open questions for the human

1. `MOTION_MIN = 0.01` is chosen from our clips; the margin is a noise margin
   (fine at sensor sigma 4/255, collapsed at sigma 8). Re-measure on the real
   camera, especially in the darker environment still to be tested.
2. Should camera motion invalidate an existing lock, or just suspend acquisition?
3. Is the ~1 s acquisition delay for a held-still hand acceptable, or should the
   motion gate also accept "present and verified for N consecutive frames"?
4. Is 0.10 the right acquisition gate, or should it go lower (0.08 measured
   identically, 0.06 slightly worse on hand frames)?
