# Motion-blob sensitivity + blob-specific verifier (v9)

Two changes, done together because sensitivity alone would make the moving-clutter
problem worse without fixing its cause:

1. **Sensitivity**: the blob proposal path now runs at a lower pixel threshold
   paired with a larger area floor, which recovers slow/subtle motion *and* takes
   the static-room false-blob rate to zero.
2. **A blob-pipeline-specific verifier**: blob crops are a different distribution
   from the crops the main verifier was trained on, so a second model is trained
   on crops built through the blob pipeline itself, at its own threshold.

Everything is still behind `--motion-blobs`, **off by default**.

## Step 1: sensitivity, and what it costs

`sweep_motion_sensitivity.py` sweeps the three levers independently in one pass per
clip (a 9-frame ring lets every ring length be evaluated from the same frames, one
difference image serves every pixel threshold):

| config (span 5 = shipped ring) | CloseLightStill (tremor) | VeryLowLight | NohandLight (noise) |
|---|---|---|---|
| shipped: thr 12, area 0.03 | 36.8% | 27.0% | 2.7% |
| thr 6, area 0.03 | 45.8% | 36.0% | **10.8%** |
| thr 12, area 0.05 | 36.3% | 14.5% | 0.0% |
| **thr 6, area 0.05 (shipped in v9)** | **45.8%** | **31.5%** | **0.0%** |
| thr 3, area 0.05 | 25.9% | 36.5% | 0.0% |

* **The pixel threshold alone is the wrong lever.** Lowering it to 6 gains ~9
  points of slow-motion detection but quadruples the static-room false-blob rate
  (2.7% -> 10.8%). Lowering it to 3 is worse still for detection *and* catastrophic
  for noise at the old area floor.
* **The area floor is what decouples the two.** Noise speckle clusters are small; a
  slowly moving hand is not. At 0.05 the false-blob rate is 0.0% even at thr 6.
* Lengthening the ring was rejected: it helps the held-still case but pushes the
  fast-motion blob centroid *behind* the hand (fast-motion on-hand rate 22.3% ->
  8.4%), which is a localisation regression, not a win.

`MOTION_PIXEL_THR` (the gate's own metric, validated separately in v5/v6) is
**unchanged** at 12; the blob path gets its own `MOTION_BLOB_PIXEL_THR = 6`.

**Noise margin (`noise_margin_blobs.py`), re-measured rather than assumed:**

| config | held-still on-hand, sigma 0..8 | static-room false blobs, sigma 0..8 | breaks at |
|---|---|---|---|
| gate thr 12, area 0.03 (v5-v8) | 32.7-37.2% | 0-2.2% | > sigma 12 |
| blob thr 6, area 0.03 | 41.8-44.9% | 8.9-13.3% | sigma ~12 |
| **blob thr 6, area 0.05 (v9)** | **41.8-44.9%** | **0.0%** | sigma ~12 |
| blob thr 12, area 0.05 | 32.7-37.2% | 0.0% | > sigma 12 |

So the new blob threshold keeps its full detection gain through sigma 8 and breaks
at sigma ~12 (48.9% false blobs on NohandLight) versus > 12 before. Measured
sensor noise on the real footage is sigma ~ 1.05, so the margin is roughly **8x the
real footage's noise**, down from ~12x. That is a real reduction.

**Cost (Step 1.3):**
* moving-clutter test: unchanged in kind - still 100% of locks on the moving
  object (6/6 before, 6/6 at the new sensitivity);
* 7-clip regression with blobs ON: **identical to v6** (469 hand frames / 15 clutter
  locks; NohandLight and NohandDark at 0 acquisitions, 0 accepted frames).

## Step 2: why the verifier could not discriminate blob crops

`diagnose_blob_crops.py`, same frame, same hand, both preprocessing pipelines:

| crop pipeline | n | median P(hand) | >= 0.30 |
|---|---|---|---|
| blob crop, ON hand (max of 8) | 88 | 0.395 | 75.0% |
| blob crop, ON hand (single rotation) | 88 | 0.249 | 39.8% |
| **training-style crop, same hands** | 41 | **0.203** | 22.0% |
| blob crop, OFF hand (max of 8) | 297 | 0.409 | 75.8% |
| blob crop, OFF hand (single rotation) | 297 | 0.235 | 33.7% |

Three findings:
1. **Preprocessing mismatch is real**: for the SAME hands, blob-pipeline crops score
   **1.95x** higher than the training-style crops the verifier was trained on. The
   crop statistics are similar (mean 35 vs 39, contrast 0.39 vs 0.32), so this is
   geometric/structural, not brightness.
2. **The 8-way max search inflates clutter more than hands** (+0.175 vs +0.146),
   which is enough on its own to flip the separation negative (75.0% vs 75.8%).
3. With a single orientation the separation is positive but tiny (0.249 vs 0.235).

**Fix (`blob_verifier.py`)**: a second SmallCNN trained on crops built through the
*actual* blob pipeline - same ROI scale, all 8 orientations as separate samples
sharing the blob's label - labelled by the MediaPipe arbiter, with off-hand blobs
from five clips plus 160 injected-moving-clutter crops as hard negatives. Held-out
blob frames only (every 3rd block of 50).

| model / rule | on-hand accepted | off-hand accepted | separation |
|---|---|---|---|
| *before*: main verifier, thr 0.30 (v8 measurement) | 75.0% | 75.8% | -0.8 |
| main verifier on these crops, thr 0.30, max | 22.6% | 9.5% | +13.1 |
| **blob verifier, thr 0.30, max** | 91.9% | 54.7% | **+37.2** |
| blob verifier, thr 0.30, mean | 90.3% | 47.5% | +42.8 |
| blob verifier, thr 0.30, single | 90.3% | 46.9% | +43.4 |

The separation flips from **-0.8 (and -17.8..-34 in the earlier calibration) to
+37..+43**. But separation is not what decides pipeline behaviour - *absolute*
clutter acceptance is, and 54.7% is far too permissive: at thr 0.30 the pipeline
locked onto the injected moving object on **35 of 45 frames**. So the threshold was
swept (`sweep_blob_threshold.py`):

| thr | hand recall (max) | moving-clutter acceptance (max) |
|---|---|---|
| 0.30 | 91.9% | 54.7% |
| 0.50 | 87.1% | 32.4% |
| 0.70 | 80.6% | 11.2% |
| **0.80 (shipped)** | **75.8%** | **5.6%** |
| 0.90 | 74.2% | 3.4% |
| *main verifier @ 0.30, for reference* | *22.6%* | *9.5%* |

At 0.80 the blob verifier delivers **3.3x the hand recall of the main verifier at
lower clutter acceptance**. `MOTION_BLOB_THRESHOLD = 0.80` is shipped for the blob
verifier only; the main verifier keeps 0.30 for detector crops.

## Step 3: combined validation

**VeryLowLight, pipeline level** (`eval_vll_pipeline.py --motion-blobs`, held-out
split marked):

| config | accepted | on the hand | clutter locks | held-out on-hand |
|---|---|---|---|---|
| v6/v7, blobs OFF (default verifier) | 130 (20.0%) | 32 (4.9%) | 98 | **7 / 200** |
| v6/v7, blobs OFF (VLL verifier) | 177 (27.2%) | 66 (10.1%) | 111 | 8 / 200 |
| v8, blobs ON, **no** blob verifier | 455 (69.9%) | 50 (7.7%) | **405** | 10 / 200 |
| **v9, blobs ON + blob verifier @0.80** (default verifier) | 204 (31.3%) | 59 (9.1%) | 145 | **9 / 200** |
| v9, blobs ON + blob verifier (VLL verifier) | 200 (30.7%) | 76 (11.7%) | 124 | 8 / 200 |

The blob verifier removes most of v8's clutter cost: clutter locks **405 -> 145**
(-64%) while on-hand positions stay level or improve, and against the *pre-blob*
baseline on-hand positions are up **84%** (32 -> 59) at a 48% increase in clutter
locks (98 -> 145). Held-out: 7 -> 9 frames of 200.

**MediaPipe on the same clip: 44.7% static / 60.8% tracking.** We are at 4.5% of
held-out frames. **We still do not beat MediaPipe.**

**7-clip regression** (blobs ON + blob verifier): **identical to v6** - 469 hand
frames / 15 clutter locks, and NohandLight / NohandDark at **0 acquisitions, 0
accepted frames**. No regression.

**Moving-clutter test** (synthetic: `NohandLight` + a translating scene-texture
patch, with acquisition attribution):

| config | acquisitions | from blob | accepted frames | on the object |
|---|---|---|---|---|
| blobs OFF (v6/v7) | 0 | - | 0 | - |
| **blobs ON + blob verifier** | **1** | 1 | 22 | **22 (100%)** |

Still one acquisition, now on the moving non-hand object, which TRACK then holds
and follows for 22 frames. The blob verifier reduced the *rate* (35 -> 22 frames)
but not the outcome. Caveat unchanged: the object is a synthetic translating patch,
not a real curtain/fan/person, and no recorded clip contains real moving clutter.

## Second look

* **Is the sensitivity increase alone usable?** No. At thr 6 / area 0.03 it already
  quadruples static-room false blobs (2.7% -> 10.8%), and with no blob verifier the
  mode locked onto moving clutter 6/6 times. Sensitivity raises how often blobs are
  proposed, so shipping it alone would make the v8 failure mode *more* frequent.
  The area floor keeps the noise cost at zero but does nothing about moving clutter
  - that is entirely the verifier's job, and it is only partly fixed.
* **Does the blob-specific verifier generalise, or is it tuned to this room/lamp/
  camera?** Better than the v7 retrain (it trains across five clips rather than one)
  but still one camera: held-out blocks are from the same clips, and there is no
  independent room. The 7-clip regression being bit-identical is evidence of no
  *damage* elsewhere, not evidence of cross-condition generalisation.
* **Does training on blob-pipeline crops risk learning the orientation-search
  process?** Measured: at thr 0.80, max-of-8 = 75.8% / 5.6%, mean-of-8 = 74.2% /
  3.9%, single-orientation = 72.6% / 3.9%. All three are within ~3 points on
  recall and ~2 on clutter, so the model is not exploiting the search; the max rule
  is simply a slightly more generous decision rule. (In v8 the max rule *did* favour
  clutter, but that was the main verifier being out of domain on blob crops - the
  effect disappeared once the model was trained on them.)

## Recommendation: keep it as an A/B option, not the default

Step 2 found and fixed a genuine, measurable cause, and the crop-level result is
strong. But the mode still fails the failure-mode test that matters: **one
acquisition on a moving non-hand object is enough to take and hold the cursor for
22 frames**, and on the target clip it buys +84% on-hand positions at +48% clutter
locks while remaining at 4.5% held-out recall against MediaPipe's 44.7-60.8%.

The order of what is left, by size: the detector's localisation (64.3% of hand
frames lost in this lighting) is still the binding constraint, and no amount of
blob-proposal work touches it. Ship as `--motion-blobs` for live A/B; do not make
it default until the moving-clutter outcome changes.
