# VeryLowLight diagnosis, dataset and retrain (v7)

The target condition: `VeryLowLight.mp4` (640x480, 15 fps, 651 frames, 43 s) --
the user's hand moving on screen under artificial light in a slightly dark room.
Both our pipeline and the MediaPipe baseline "struggled badly" live.  This records
what was measured, what was built, what worked, and what did not.

## Step 1 - where the failure actually is

Tools: `diagnose_verylowlight.py` (full stage funnel, per-frame CSV in
`build/verylowlight_perframe.csv`), `gt_verylowlight.py` (MediaPipe truth, in the
mediapipe venv), `arbiter_vll.py` (independent adjudicator).

Measured properties of the clip:

| property | VeryLowLight | other clips |
|---|---|---|
| frame luma | 43.5 (34.6 .. 66.2, drifts mid-clip) | 29-146 |
| contrast (std) | ~17 | 36-53 |
| colour cast R/B | **1.79** | 1.05-1.21 |
| sensor noise sigma | **1.05** | 0.0-1.05 |

So: the artificial light has a strong warm cast (R/B 1.79 against 1.05-1.21
everywhere else) and about a third of the usual contrast.  **The sensor-noise
worry from `motion_gate.md` is NOT realised here**: sigma ~ 1.05 against
the ~8 where the motion gate's separation collapses.

MediaPipe on the same frames: detects the hand in **60.8%** (tracking mode) /
44.7% (static mode, which has no temporal state and therefore no lag).  So
MediaPipe is degraded, not blind.

Stage funnel over all 651 frames (v6 chain, MediaPipe static as truth):

| stage | frames | of all |
|---|---|---|
| palm detector fired (score >= 0.5) | 389 | 59.8% |
| + size gate (0.10) | 373 | 57.3% |
| + motion gate | 227 | 34.9% |
| + CNN verifier (>= 0.30) | 68 | 10.4% |

Attribution over the **291 frames where MediaPipe (lag-free) sees a hand**:

| what happened | frames | share |
|---|---|---|
| detector did not fire at all | 77 | 26.5% |
| detector fired, but not on the hand | 110 | 37.8% |
| size gate rejected | 1 | 0.3% |
| **motion gate rejected** | 17 | **5.8%** |
| **CNN verifier rejected** | 65 | **22.3%** |
| survived the whole chain | 21 | 7.2% |

Supporting measurements:

* Detector confidence is **not** marginal: the count of frames with a detection is
  identical (59.8%) at every score threshold from 0.3 to 0.7, best score median
  0.760.  It fires confidently - just usually not where the hand is.  Offset from
  MediaPipe's palm centre is bimodal: 37% of detections are within 0.35 box widths
  (dead on), median 1.47, 76% within 2.
* Motion gate: 9.7% of hand detections fall below the motion floor; on hand
  detections the median motion frac is 0.191 against a 0.01 floor.  The gate is
  not the problem, and the global-change guard fires on 3 frames (0.5%).
* CNN verifier: on crops whose box contains the hand, median P(hand) = **0.128**,
  and 79.6% fall below the 0.30 threshold.

**Independent arbiter.** The landmark model (a different network, answering "is
this crop a hand?") was run on three crop sets per frame, using the pipeline's own
rotation recipe.  Anchored on MediaPipe's own 21 landmarks - i.e. at the exact
spot MediaPipe says the hand is - presence is **0.000 (median, p90 0.001, 0% above
0.5) over 291 frames**.  Controls with the identical recipe:

| clip | presence on our detections (median) | > 0.5 |
|---|---|---|
| CloseLightStill (daylight, close) | 1.000 | 100% |
| CloseDarkStill (dark, close) | 0.192 | 24.7% |
| FarLight (daylight, far) | 0.008 | 0.9% |
| **VeryLowLight (artificial light, close)** | **0.001** | **3.8%** |

A close-range hand that the landmark model cannot see at all, at a location
MediaPipe asserts contains a hand, while the same recipe scores 1.000 in daylight:
that is a domain shift affecting the model family, not a crop-recipe artefact and
not detector confusion alone.

**Diagnosis.** Three things fail, in this order of size:
1. the palm detector's *localisation* (64.3% of hand frames lost: 26.5% no
   detection, 37.8% detection off the hand);
2. the CNN verifier (22.3% lost, median P(hand)=0.128 on genuine hand crops);
3. the motion gate (5.8%, and the noise-margin concern is not realised).

So more verifier data cannot fix this on its own - the detector is upstream of it,
and its weights/architecture are off-limits by standing instruction.

## Step 2 - data built (informed by Step 1)

`build_vll_dataset.py` -> `vll_dataset/` (32027 crops):

* **Real crops from the target clip** (286 MediaPipe-anchored positives using
  MediaPipe's own landmarks with the pipeline's rotation recipe, 133
  detector-recipe positives, 218 clutter negatives from frames where neither
  MediaPipe mode sees a hand near the box).  Split by *block* (every 3rd block of
  50 frames held out, ~1/3 of the clip) so both halves span the clip's brightness
  drift and adjacent frames cannot leak across the split.
* **Synthetic augmentation of the existing set**: only the *training* sources are
  augmented (`hagrid_train`, `own_clip`); `hagrid_valid`/`hagrid_test`/`own_nohand`
  are carried across untouched and stay held out.  The photometric transform is
  **measured, not guessed**: per-channel mean/std matching to VeryLowLight's
  statistics (mean BGR [32.9 38.7 57.4], std [19.3 24.0 34.0]), which reproduces
  the darkening, the warm cast and the contrast loss in one step.  Three
  severities (0.5/0.8/1.0), target statistics jittered +-15% per sample and
  severity jittered +-0.1 so the model cannot lock onto one colour temperature,
  plus random downscale-upscale (0.2-1.0) for range.  Applied to **both** classes.

Composition: 20187 augmented + 11203 originals carried across + 637 real VLL
crops.  Real data is limited exactly as Step 1 predicts - the detector gives us
hand-containing boxes in only ~36% of hand frames, which is why augmentation of
the existing set carries most of the weight.

## Step 3 - retrain and evaluate

`train_vll_verifier.py` -> `verifier_cnn_vll.pt` (same SmallCNN, 8 epochs,
21508 balanced crops).  Held-out results, previous vs retrained:

| split | previous | retrained |
|---|---|---|
| VLL held-out hands (MediaPipe-anchored, n=83) kept | 22.9% | **59.0%** |
| VLL held-out hands (our detector recipe, n=41) kept | 4.9% | **65.9%** |
| VLL held-out clutter (n=72) cut | 93.1% | 90.3% |
| HaGRID held-out (valid+test, n=4451) AUC | 0.9650 | 0.9682 |
| HaGRID kept / cut | 87.2 / 93.1% | 94.3 / 85.4% |
| own-room clutter (n=23) cut | 91.3% | 87.0% |

At the crop level that is a large, real gain in the target domain (2.6x and 13x on
the two positive sets) with daylight generalisation intact - it is a recall-leaning
trade (clutter cut 93.1% -> 85.4% on HaGRID).

**But it does not translate into the pipeline.**  Full v6 chain replayed over the
whole clip (`eval_vll_pipeline.py`), scored against MediaPipe, with the held-out
split marked:

| | previous verifier | retrained |
|---|---|---|
| accepted positions (all frames) | 130 (20.0%) | 177 (27.2%) |
| of those, ON the hand | 32 (4.9%) | 66 (10.1%) |
| clutter locks | 98 | 111 |
| held-out frames with a position on the hand | 7 / 200 (3.5%) | **8 / 200 (4.0%)** |

**MediaPipe on the same clip: 44.7% (static) / 60.8% (tracking).**

So we do **not** beat MediaPipe on this footage, and the retrain barely moves the
held-out pipeline number (7 -> 8 frames of 200).  The gain on all frames is
inflated by training blocks.  The reason is simply that the pipeline never gets a
hand-containing crop to classify most of the time: in REACQUIRE the verifier is not
applied at all, and the TRACK window is anchored on the detector's own box, which
usually is not on the hand.

**It also regresses the clips already validated.**  Same v6 pipeline over the
original 7 clips, only the verifier swapped:

| verifier | hand frames | clutter locks |
|---|---|---|
| verifier_cnn.pt (current default) | 469 | 15 |
| verifier_cnn_vll.pt (retrained) | 460 | **37** |

FarLight in particular goes 6 -> 29 clutter locks.  **Therefore the retrained
model is bundled but NOT the default**; switch it in live with
`--verifier verifier_cnn_vll.pt` to judge it in the real dark room.

## Second look

* **Does the retrained verifier generalise?** Partly.  It generalises across
  *frames* (held-out blocks of the same clip: 22.9% -> 59.0% of hand crops kept),
  which is the measure of "not memorising individual frames" that I trust.  It does not
  generalise across *conditions*: all held-out crops come from the same room, lamp
  and camera, so this is "this lighting", not "artificial light".
* **Is the colour-cast augmentation overfit to one colour temperature?**  Partly,
  measured in `critic_vll_cast.py` by re-lighting held-out crops:

  | condition | prev kept | new kept | prev cut | new cut |
  |---|---|---|---|---|
  | as recorded (R/B 1.79) | 16.9% | **61.3%** | 93.1% | 90.3% |
  | warmer (R/B x2.0) | 6.5% | 38.7% | 98.6% | 97.2% |
  | cooler (R/B x0.8) | 13.7% | 52.4% | 75.0% | 95.8% |
  | near-neutral (R/B x0.6) | 1.6% | **4.0%** | 90.3% | 97.2% |
  | brighter (x1.6) | 31.5% | 58.1% | 93.1% | 90.3% |
  | darker (x0.6) | 30.6% | 27.4% | 69.4% | 100.0% |

  Shifts of +-20-40% in cast keep 39-60% of hand crops, so it degrades gracefully
  rather than shattering - but removing the cast entirely collapses it to 4%.
  The model has learned this cast as part of "what a hand looks like here".  The
  per-sample jitter bought robustness, not independence.
* **Headline:** we still do not beat MediaPipe on this footage
  (4% vs 45-61% usable frames).  The upstream detector limitation dominates, and
  retraining the verifier alone was not enough.

## What would actually be needed (not done)

1. Fix detector localisation in this lighting - the single biggest loss (64% of
   hand frames).  Currently off-limits (weights/architecture untouched by
   instruction), so this needs an explicit decision to revisit.
2. Apply the verifier during REACQUIRE too, so a hand-containing detection can be
   accepted before a lock exists (today REACQUIRE is size+motion only).
3. More real dark footage: 286 positives from one clip is thin, and they all share
   one lamp, one room and one camera.
