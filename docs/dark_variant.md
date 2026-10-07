# Dark-specialised variant (HandCursorDark_v1) - experimental

**Separate from v10 by design. `HandCursor_v10` remains the safe default and is
unchanged.** This variant is allowed to trade light-condition precision for
dark-condition recall, and it does.

## Step 0 - status of the earlier motion-sensitivity task: it LANDED

Verified in the codebase rather than assumed:

| item | state |
|---|---|
| lowered blob pixel threshold (`MOTION_BLOB_PIXEL_THR = 6.0`) | **in code** |
| raised blob area floor (`MOTION_BLOB_MIN_SIDE_FRAC = 0.05`, was 0.03) | **in code** |
| separate difference image for blobs (`_ensure_diff(blob=True)` -> `_blob_diff`) | **in code** |
| blob-specific verifier (`blob_verifier.pt`, `MOTION_BLOB_THRESHOLD = 0.80`) | **trained, wired, bundled in v10** |
| validation evidence | `build/sweep_sensitivity.txt`, `noise_margin_blobs.txt`, `blob_verifier.txt`, `eval_oldclips_v9.txt`, `test_moving_clutter.txt` |
| docs | `motion_blobs.md`, `motion_blobs_v9.md` |
| blob path default | still **off** (`--motion-blobs`) |

The only casualty of that round was **its build**: `HandCursor_v9` crashed at
startup in the default configuration and was never reported as failed until the
next round. It was fixed and superseded by v10, and `test_startup.py` now guards
the whole class of failure. Nothing about the underlying work was abandoned, and
Step 3 below builds on it directly.

## Step 1 - dark/distance-heavy augmentation (no new raw data)

`build_dark_dataset.py` on top of the **arbiter-confirmed** crop pool (so heavier
augmentation multiplies good labels, not the ~42% of mined crops whose boxes were
not on the hand - that fix is upstream in `big_dataset` and is inherited here).

Variants per source crop: four photometric severities of the measured match to
VeryLowLight's statistics (0.35 / 0.55 / 0.75 / 1.0), three distance bands
(none / mild 0.45-1.0 / strong 0.12-0.35) drawn **independently**, so most samples
are dark *and* downscaled - the actual target regime. Identical procedure applied
to both classes, so brightness and sharpness cannot become class cues.

| | count |
|---|---|
| full dark dataset | **600,500 crops** (595,830 trainable + 4,670 held out) |
| subsampled for training | 84,670 (43,587 hand / 41,083 clutter) |
| held-out splits | unchanged from every previous round |

Held-out results, `verifier_dark.pt` vs the previous default:

| split | `verifier_cnn.pt` | `verifier_dark.pt` |
|---|---|---|
| VLL held-out hands kept (MediaPipe-anchored) | 22.9% | **55.4%** |
| VLL held-out hands kept (detector recipe) | 4.9% | **56.1%** |
| VLL held-out clutter cut | 93.1% | 81.9% |
| HaGRID held-out AUC | 0.9650 | 0.9043 |
| own-room clutter cut (n=23) | 91.3% | 73.9% |

At **matched own-room precision** (threshold 0.70) it keeps 47.0% / 46.3% of VLL
hand crops against the 11k reference's 41.0% / 4.9% - a real gain, **but it still
loses to `verifier_cnn_big.pt` @0.61 (57.8% / 58.5%)**. So the heavy dark
augmentation did not beat the volume model: the dark variant ships the big model
and `verifier_dark.pt` stays available as an alternative. Recorded rather than
quietly dropped.

## Step 2 - detector fine-tune (the main effort)

**Design** (`finetune_detector_dark.py`, written before training):

* **Unfrozen: the six 1x1 prediction heads only** - 43,966 of 1,763,358 params
  (**2.49%**). Everything else frozen (backbone1/2/3, both scale paths,
  `scaled16add`). A domain shift can be absorbed as a per-anchor offset/scale
  without touching the features the rest of the pipeline depends on, and with ~200
  real labelled dark frames the smallest parameter set is the only safe one.
  Spatial detail has already collapsed by the 8x8 stage, so unfreezing deeper
  buys nothing. `--unfreeze heads_last` adds `scaled32add` for comparison.
* **Data:** VeryLowLight **train blocks only** - 203 labelled frames - with the
  Step 1 dark/distance augmentation (25% left unaugmented so the head does not
  forget the light regime entirely).
* **Held out:** VeryLowLight holdout blocks (88 frames) **and all seven original
  clips**, which is what makes the light-condition trade a real one rather than
  in-sample. Generalisation risk stated up front: one room, one lamp, one camera.
* **Losses:** BCE over all 2944 anchors (pos_weight 5) plus smooth-L1 on the box
  and on keypoints 0 (wrist) and 2 (middle MCP) for the matched anchors. Only
  those two keypoints are supervised - they are the only ones the port's ROI code
  consumes and the only ones MediaPipe's landmarks map onto unambiguously; the
  other ten keep their pretrained values.
* **Labels:** target encoding verified as the exact inverse of the port's own
  `_decode_boxes` - **round-trip error 0.0000-0.0001 px**.

**First attempt failed, and the failure is worth recording.** The naive matching
rule ("every anchor whose centre is inside the box") labels ~96 of 2944 anchors
positive for a 2.6x-palm box; with pos_weight 20 that diffuse, heavily weighted
signal taught the score head to fire everywhere - anchors >= 0.5 went from 13 to
**337** on a light frame, and containment collapsed to 0% on every clip. Diagnosed
by looking at score sparsity, fixed with central-anchor matching (<= 6 anchors),
pos_weight 5 and lr 3e-4, and a **collapse guard** now refuses to save a model with
more than 100 hot anchors. (The port's anchors are all unit-size, so MediaPipe's
scale-aware matching is not available; keeping the positive set small and central
is the available lever.)

**Result - the light/dark trade, measured on held-out data:**

| case | original | fine-tuned | delta |
|---|---|---|---|
| **VeryLowLight holdout (as recorded)** | 25.0% | **33.0%** | **+8.0** |
| VeryLowLight holdout + dark aug | 25.0% | 39.8% | +14.8 |
| VeryLowLight train blocks (in-sample) | 29.2% | 41.2% | +12.0 |
| CloseLightMoving | 98.9% | 89.1% | −9.8 |
| CloseLightStill | 100.0% | 100.0% | 0.0 |
| CloseDarkStill | 100.0% | 93.6% | −6.4 |
| FarLight | 11.2% | 22.5% | **+11.2** |
| FarDark | 32.8% | 40.2% | **+7.4** |
| **seven light clips, mean** | | | **+0.4** |

Overfitting gap (train-block minus holdout, dark-augmented): **+4.2 → +1.5 points**  - 
the fine-tuned model generalises *better* than the original on this split, because
the original is uniformly poor in the dark. Original weights untouched:
`palmdetector.pth` is byte-identical, the fine-tune is saved as
`palmdetector_dark.pth`.

## Step 3 - more motion weight, dark variant only

The dark entry point (`hand_mouse_dark.py`) runs with **`--motion-blobs` on by
default**, plus the blob-specific verifier at 0.80. Rationale as agreed: in very
low light a user will naturally wave, so leaning on motion is a deliberate trade
for this variant. v10's default path is untouched.

## Step 4 - three-way comparison (`build/DARK_summary.txt`, `FINAL_summary_default.txt`)

| clip | MediaPipe | **v10 default** | **dark variant** |
|---|---|---|---|
| CloseLightMoving | 100.0% | 85.7% | **94.3%** |
| CloseLightStill | 100.0% | 92.5% | 54.1% |
| CloseDarkStill | 99.5% | 93.8% | **12.1%** |
| FarLight | 74.6% | 32.3% | **52.6%** |
| FarDark | 77.8% | 14.9% | **51.7%** |
| VeryLowLight | 65.1% | 8.8% | **31.8%** |
| NohandLight false locks | 0.0% | 0.0% | **2.2%** |
| NohandDark false locks | 0.0% | 0.0% | 0.0% |
| **aggregate F1** | **81.5%** | 50.1% | 49.7% |
| aggregate precision / recall | 88.3 / 75.6 | 81.3 / 36.2 | 63.4 / 40.9 |

F1 per clip. The dark variant **more than triples** VeryLowLight (8.8 → 31.8),
**triples** FarDark (14.9 → 51.7) and improves FarLight (32.3 → 52.6) - the three
conditions the project targets - while giving up CloseDarkStill (93.8 → 12.1) and
CloseLightStill (92.5 → 54.1).

**Ablation** (same detector and verifier, motion blobs **off**): NohandLight false
locks **26.7%**, aggregate no-hand 12.3%, CloseDarkStill 6.2%, VeryLowLight 32.9%.
So blobs-ON *reduces* the no-hand regression five-fold - because that path
verifier-gates REACQUIRE - which identifies the fine-tuned detector as the source
of the false locks, not the motion path.

## Second look

* **Does the fine-tune generalise, or memorise one room?** Evidence for genuine
  transfer: +8.0 points on *held-out blocks of the same clip*, +14.8 in the
  training regime, and - most tellingly - **+11.2 and +7.4 on FarLight/FarDark,
  which are daylight clips the model never saw**. The distance augmentation
  evidently taught something about small-hand detection that transfers across
  lighting. The overfit gap also shrank (+4.2 → +1.5). Against that: it loses 6-10
  points on the close clips, and it introduced false locks on a no-hand clip. So
  partial cross-condition transfer (distance yes; close-range and static-room
  behaviour degraded), still from one room/lamp/camera of real dark data.
* **Is the trade proportionate?** In the target conditions, yes and clearly:
  +23 F1 on VeryLowLight, +37 FarDark, +20 FarLight. But it is a *redistribution*,
  not a free gain: CloseDarkStill loses 82 points, and aggregate F1 is a wash
  (49.7 vs 50.1). The variant is for far/dark *moving* hands; it is worse than v10
  for close hands and for hands held still.
* **No-hand regression: FLAGGED.** 0.0% → 2.2% on NohandLight (26.7% with blobs
  off). This is the first non-zero false-lock rate in the project's history and the
  reason this build is not the default. Cause: the fine-tuned detector; the blob
  path masks most of it by verifier-gating REACQUIRE. NohandDark remains 0.0%.

## Recommendation

Keep `HandCursor_v10` as the default. Use `HandCursorDark_v4` (experimental) when
working at distance or in dim/artificial light with a moving hand, where it is
2-3x better than v10 - and expect it to be worse close-up and for still hands, and
to false-lock occasionally in a static room.

v3 added multi-pass spatial consensus on REACQUIRE. **v4 adds "if it is not certain,
it should not move the mouse there"**: nothing is published until a REACQUIRE run is
confirmed, and an in-window TRACK candidate that jumps more than 0.06 frame widths
from the committed position is held - the cursor stays exactly still - until three
consecutive candidates agree on the new place, or six frames pass. Measured effect:
off-hand cursor placements 362 → 164, wrong-location excursions **95 → 39**, and the
first clean run on NohandLight (false locks 2.2 % → 0.0 %); paid for with recall
(F1 49.8 → 46.4) and a regression on CloseDarkStill (16.0 → 6.2), where the *true*
detections are the transient ones. Full design and measurements in
`consensus.md`; `--track-consensus-tol 0.09` and `--track-consensus 0` are the
one-flag escape hatches.



---

## HandCursorDark_v1 was broken - what happened (and what now prevents it)

`HandCursorDark_v1.exe` crashed about a second after the camera opened. This was a
second instance of the same class of failure as the v9 `UnboundLocalError`: a real
defect that every check on the build machine passed, because the checks were run in
conditions the user never encounters.

**Root cause, in layers.** The dark build asks for
`--verifier verifier_cnn_big.pt` - a *bare* filename. Model lookup tried the name as
given (i.e. relative to the process cwd), then the bundled `models` folder. That
works when the app is launched from the project root, where the file is also
reachable, and it fails on a **double-click**, whose cwd is the exe's own folder.
The verifier therefore went inactive without any complaint - the log politely said
`Verifier NOT active - TRACK uses the 0.04 box-size gate (v2 behaviour)`. In v10
that degradation is survivable. In the dark build motion blobs are ON by default,
and the blob path handed the `None` verifier to rotation scoring:

```
TypeError: 'NoneType' object is not callable
```

on roughly frame 5 - the "crashes in about a second" the user saw. Reproduced
headlessly before fixing anything.

**Why the checks missed it, twice.** (1) `build_release.py`'s sanity check ran the
exe with cwd = the *project root*, where the bare filename resolves fine - so the
one condition that mattered was never exercised. (2) It also ran on a machine with
no webcam, so the app printed `could not open webcam` and exited *before* the loop
where the crash lived; "reached the webcam step" was treated as success. Neither
check could see the bug by construction. The earlier v9 defect hid the same way:
every smoke test that round passed `--motion-blobs`, so the default path was never
started.

**Fixes, in the app.**

* `resolve_model_path(name)` - one resolver for every model flag: as-given, then
  the bundled `models` dir, then the script dir. Used by `--detector`,
  `--verifier`, `--blob-verifier` and the defaults.
* An **explicitly requested** model that cannot be found is now **fatal**
  (`SystemExit` with the searched paths) instead of silently degrading. Degrading
  quietly is what turned a config error into a crash a second later.
* The blob path no longer assumes a verifier exists: `blob_candidates` returns
  `[]` and the locator falls back to plain non-blob selection.
* If no verifier at all is available, `--motion-blobs` is refused with a printed
  warning (unconfirmed blobs are the measured-unsafe mode: 88-94% of OFF-hand
  motion blobs accepted).
* New `--no-motion-blobs`, so the dark build's forced-on blob path can be turned
  off from the command line. `hand_mouse_dark.py` respects it.

**Fixes, in the checks.**

* `test_startup.py` (permanent part of the suite) now has four sections: source
  configurations *including the zero-argument double-click path*; the locator loop
  run **headlessly over 40 real frames** - including the exact crash configuration
  (blobs on, verifier `None`) - so no camera is needed to exercise it; bare model
  names resolved from a foreign working directory; and every frozen build launched
  **with zero arguments and cwd = the exe's own directory**, which is what a
  double-click actually does. `HandCursorDark_v1` is kept as a **canary**: the
  suite asserts it *still* reports "verifier NOT loaded", so if the test ever stops
  detecting the known-bad build, that itself fails.
* `build_release.py`'s sanity check now launches with cwd = the exe's directory,
  requires the verifier to load for **both** `hand_mouse_cursor.py` *and*
  `hand_mouse_dark.py` (it previously only expected one for the main entry, so the
  dark build could pass with no verifier), and prints a clear note when there is no
  webcam, so "reached the webcam step" is never mistaken for "the loop works".

**Rebuild.** The fix ships as `HandCursorDark_v2`; `v1` is left in `dist/`
untouched and marked broken. The dark variant's measured behaviour in the tables
above was produced by the same code and models, so those numbers still apply  - 
the crash was in model *resolution*, not in the detector, verifier or blob logic.

