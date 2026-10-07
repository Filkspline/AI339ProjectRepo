# Verifier v2: real HaGRID volume, an arbiter, and what it actually bought (v10)

Final push on two fronts: train on the real HaGRID volume instead of the small
subsample, and replace "vibes" evaluation with a re-runnable harness.  Headline:
**more real data helps the target condition, but not enough to become the default.**

## Part 1 - the real HaGRID volume

### What was used before (confirmed on disk)

Exactly **1.18 GB from `zhouxzh/portable-hagridv2-mediapipe-hand`** (4 parquet
shards), which is what produced the 11,203-crop `verifier_dataset`. The 10 GB
subset was **never** pulled. Those shards carry palm boxes and keypoints, so crops
could be mined by geometry.

### What was pulled now

`testdummyvt/hagRIDv2_512px` - the real 512px HaGRIDv2, **64.2 GB total as 1503
train + 301 test shards**. Pulled **10.03 GB / 157 shards** (~10% of train) with
`fetch_hagrid_bulk.py`, which is resumable (skips complete shards, retries per
file) and took 44.5 min at ~4.5 MB/s.

Those shards carry **only `image` (512px JPEG) + `label`** (34 classes, **13 =
no_gesture**). No bounding boxes - so crops must be mined with the detector rather
than by geometry, and `no_gesture` images are skipped (per the dataset card they
are "natural hand postures", i.e. they contain hands: neither clean positives nor
clutter).

### The finding that mattered: the mining labels were 42% wrong

Mining a positive as "the detector's top detection on an image labelled with a
gesture" conflates *this image contains a hand* with *this detection is the hand* --
and "the detector fires confidently but off the hand" is the exact failure mode
diagnosed in `verylowlight_diagnosis.md`.

Checked with an independent arbiter (the landmark model's presence head, which
scores 1.000 on real daylight hands and ~0.03 on clutter):

```
presence over 30,000 mined crops:  p10 0.000  p25 0.003  p50 0.983  p75 1.000
kept at presence >= 0.5: 58.4%      -> 41.6% rejected
```

Reproduced on the final run over 31,006 detections: **41.9% rejected**. The
distribution is sharply bimodal (crops are either clearly hands or clearly not),
which is what a *labeling* error looks like, not a threshold artefact.

Training on the unfiltered set produced exactly what that predicts - a
**permissive** verifier:

| split | verifier_cnn.pt (11k) | big, UNFILTERED |
|---|---|---|
| HaGRID held-out AUC | 0.9650 | **0.8081** |
| HaGRID kept / cut | 87.2 / 93.1% | 98.2 / **20.2%** |
| own-room clutter cut (n=23) | 91.3% | **39.1%** |

Kept as `verifier_cnn_big_UNFILTERED.pt` as evidence. **Fix:** mining now
arbiter-checks each crop inline (`--min-presence 0.5`) and only saves confirmed
hands, so the label source is never the detector's own output.

This is also independent corroboration, at scale, of the localisation problem:
**on clean, well-lit, gesture-focused HaGRID images the detector's top detection is
off-hand ~42% of the time** (versus 64.3% of hand frames lost in the artificial
light).

### Final training set (`big_dataset`, 130,532 crops)

| label | source | count | held out |
|---|---|---|---|
| hand | **hagrid512** (new, arbiter-confirmed) | **18,000** | 0 |
| hand | hagrid512_aug | 54,000 | 0 |
| hand | hagrid_train / _aug | 4,000 / 12,000 | 0 |
| hand | own_clip / _aug | 77 / 231 | 0 |
| hand | vll_real_mp / _aug | 286 / 609 | 83 |
| hand | vll_real_det / _aug | 133 / 276 | 41 |
| hand | hagrid_valid / hagrid_test | 1,139 / 2,324 | all |
| clutter | hagrid_train / _aug | 2,652 / 31,824 | 0 |
| clutter | vll_real_clutter / _aug | 218 / 1,752 | 72 |
| clutter | hagrid_valid / hagrid_test | 322 / 666 | all |
| clutter | own_nohand | 23 | all |

Held-out splits are unchanged from every previous round, so results stay
comparable. Augmentation is the established discipline, applied to **both**
classes: photometric mean/std matching to VeryLowLight's measured statistics
(mean BGR [32.9 38.7 57.4]), three severities with per-sample jitter, plus
downscale-upscale 0.2-1.0. Clutter (the scarce class) got 4x the copies, otherwise
a 1:1 balanced sampler would have discarded most of the new positive volume.

Trained 8 epochs on 72,748 balanced crops (36,374 / 36,374).

### Held-out results, and the operating-point trap

At the app's threshold (0.30) the new model looks like a regression. It is not
being compared fairly - its scores are calibrated differently, so the sweep
(`sweep_verifier_threshold.py`) puts both at **matched clutter rejection**:

| model | thr | own-room clutter cut | HaGRID cut | VLL hands kept (MP-anchored) | VLL hands kept (detector recipe) |
|---|---|---|---|---|---|
| verifier_cnn.pt | 0.25 | 91.3% | 91.5% | 41.0% | **4.9%** |
| verifier_cnn_big.pt | 0.61 | 91.3% | 70.9% | 57.8% | **58.5%** |

That is the real answer to "does more real HaGRID volume move the needle": **yes,
for the target condition** - 12x more crops, arbiter-clean, takes the
detector-recipe recall from 4.9% to 58.5% at the same own-room precision. But its
HaGRID-portable precision is worse (cut 91.5% -> 70.9%) and its AUC is lower
(0.9650 -> 0.9466), because the new positives come from a different, cleaner
distribution and the boundary moved toward recall.

### Pipeline-level regression (the standing bar)

`eval_motion_gate.py`, 7 clips, identical matching:

| verifier | hand frames | clutter locks |
|---|---|---|
| `verifier_cnn.pt` @0.30 (default) | 469 | **15** |
| `verifier_cnn_big.pt` @0.30 | 480 | 74 |
| **`verifier_cnn_big.pt` @0.61** (matched precision) | 473 | **45** |

FarLight carries it: 6 locks -> 37. Both no-hand clips stay at **0 acquisitions**
either way.

**Recommendation: A/B, not the default.** At its best operating point the new
model is roughly neutral on the validated clips (473 vs 469 hand frames) but
**triples the clutter locks** (15 -> 45), which is the one behaviour this project
has consistently chosen safety over recall for. It is bundled and switchable:

```
HandCursor_v10.exe --verifier verifier_cnn_big.pt --verifier-threshold 0.61
```

Worth trying live specifically in the dark room, where it is clearly better.

## Part 2 - automated evaluation harness

```
.venv-blazepalm\Scripts\python.exe evaluate_pipelines.py [--variants] [--verifier M] [--threshold T]
```

One command; writes `build/pipeline_comparison.csv` (one row per clip per
pipeline) and `build/pipeline_summary.txt` (presentable table + aggregate).

* **Ground truth for both pipelines, identically:** MediaPipe static mode,
  lag-free, per frame. The original 7 clips only had *tracking* data cached, which
  is the baseline itself - that scored 100% precision/recall against itself, so
  `gt_clips_static.py` generated static truth for them (`static:<clip>` keys).
  Truth is static-only and identical for both, with the same +-5-frame borrow rule.
* **Metrics per clip:** acceptance rate, precision, recall, F1, false-lock rate
  (no-hand clips), ms/frame.
* **MediaPipe baseline** = `mp.solutions.hands` in tracking mode, read from the
  cached per-frame runs, so the harness needs only the PyTorch environment.

### Where we stand (identical yardstick, 1319 truth frames each)

| pipeline | precision | recall | F1 | no-hand false locks |
|---|---|---|---|---|
| MediaPipe (tracking) | 88.3% | 75.6% | **81.5%** | 0.0% |
| Ours (v10 default) | 81.8% | 36.2% | **50.2%** | 0.0% |
| Ours + big verifier @0.61 | 80.0% | 38.4% | **51.9%** | 0.0% |

| clip | MediaPipe F1 | Ours (default) F1 | Ours + big F1 |
|---|---|---|---|
| CloseLightMoving | 100.0% | 86.0% | 87.2% |
| CloseLightStill | 100.0% | 92.5% | 92.5% |
| CloseDarkStill | 99.5% | 93.8% | 90.1% |
| FarLight | 74.6% | 32.3% | 34.5% |
| FarDark | 77.8% | 14.9% | 12.1% |
| VeryLowLight | 65.1% | 8.8% | **16.5%** |
| NohandLight / NohandDark | 0.0% false locks | 0.0% | 0.0% |

**Plain-English read.** MediaPipe is still ahead overall (F1 81.5% vs ~50%), and by
a lot. We are essentially level on the three close-range clips (F1 86-94% vs
99-100%) - a 6-14 point gap, not a different class of system. We lose badly at
distance and in the target condition: FarDark 14.9% vs 77.8%, VeryLowLight 8.8% vs
65.1%. The one thing we match or beat is safety: **both pipelines are at 0.0%
false locks** on the two no-hand clips, so our clutter-rejection work holds.

The framing of the low-light/distance gap: the 130k model **doubles**
VeryLowLight F1 (8.8% -> 16.5%) and nearly doubles its recall (5.3% -> 10.3%)
against the 11k baseline, so the condition that is "the actual point of the
project" is the one this data volume moved. It did not close the gap to MediaPipe
there, and the reason is unchanged from `verylowlight_diagnosis.md`: the palm detector's
localisation, not the verifier, is what limits us (64.3% of hand frames lost in
that light, and 42% arbiter-rejected detections on clean HaGRID images).

## Incidents (both mine, both fixed)

* **Destructive bug:** `build_big_dataset.py` rmtree'd its `--out` directory
  *before* reading `--mined-csv` from inside it, destroying a 178k-crop dataset and
  its filtered manifest. Inputs are now read first and only `crops/` is cleared.
  Only regenerable intermediates were lost; `hagrid512/` (9.34 GiB) was untouched.
* **Silent no-op flags:** `eval_motion_gate.py` never passed `--threshold`,
  `--motion-blobs` or `--no-blob-verifier` through to the locator, so two runs
  comparing thresholds produced identical numbers before I noticed and fixed it.
  One more thing: identical outputs across configs are a smell.

## Final diagnostic round: which stage limits the current build

Attribution re-run on `VeryLowLight.mp4` with the current chain, nested so each
stage is conditioned on the previous (the script originally printed marginal counts,
which can show identical numbers for two stages and hide the real funnel):

| stage, of 291 MediaPipe hand frames | default @0.30 | big @0.61 | big @0.61 + blobs |
|---|---|---|---|
| detector produced a hand-containing box | 104 (35.7%) | 104 (35.7%) | 104 (35.7%) |
| + size gate | 103 | 103 | 103 |
| + motion gate | 86 (29.6%) | 86 (29.6%) | 86 (29.6%) |
| + CNN verifier | **16 (5.5%)** | **73 (25.1%)** | **73 (25.1%)** |
| loss at the verifier | 70 (24.1%) | **13 (4.5%)** | 13 (4.5%) |
| loss at the detector | 187 (64.3%) | 187 (64.3%) | 187 (64.3%) |

1. **The verifier is no longer the bottleneck**: 13 of 86 genuine hand crops
   rejected at 0.61, versus 70 of 86 for the previous model.
2. **The detector is**, unchanged at 64.3% (77 frames no detection, 110 off-hand).
3. **Motion blobs do not close it, and are already wired.** REACQUIRE verifier-checks
   both detector candidates and blobs (confirmed in code: both are scored with the
   same rotation-search routine and both must clear the verifier threshold; the
   no-hand clips stayed at 0 acquisitions through v8-v10). In the 187
   detector-failure frames motion proposed the hand 21.4% of the time and the blob
   verifier confirmed 7.5% - a marginal 14 frames (4.8 points) at best, before
   paying the clutter cost that already keeps the path off by default.

**No fix was attempted**, because the diagnosis shows none is available cheaply:
the binding constraint is BlazePalm's localisation, which is off-limits by standing
instruction. Report ends here rather than chasing it further.
