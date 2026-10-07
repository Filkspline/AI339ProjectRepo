# AI339: A hand-tracking cursor for lean-back computing

This is my final project for AI339. The goal was a webcam cursor you can drive with
your hand from 2 to 4 metres away in a dim room, so you can control a screen without
sitting at a desk.

The short version of what I found: MediaPipe's own hand tracking is still clearly
better than what I built in aggregate (F1 81.5% against my 50.1%), but the interesting
part is *where* it fails and why. In my target condition the palm detector loses 64% of
the frames where a hand is actually present, before any of my code runs. I spent most of
the project building things around that detector (a learned crop verifier, a motion
gate, motion-blob proposals, agreement-over-time rules) and measuring what each one
bought. Some of them worked, some of them did not, and I documented both.

## How it works

Each frame goes through this chain:

1. **Palm detection.** BlazePalm (the detector inside MediaPipe) runs on the whole frame
   in the `REACQUIRE` state, or on a small crop around the hand's last known position in
   the `TRACK` state. Tracking inside a window is what lets me follow a hand further away
   than a full-frame detection can see.
2. **Size gate.** Rejects candidate boxes that are too small. 0.10 of the frame width
   when the motion gate is on, 0.15 when it is off.
3. **Motion gate (acquisition only).** Compares a candidate region across a 5-frame ring
   and rejects it unless enough pixels changed. Every false lock I was getting was on a
   static object, and even a hand held still has micro-tremor, so motion separates them.
4. **CNN crop verifier.** A small network I trained (60,929 parameters) that takes the
   rotation-normalised crop and returns P(hand). This replaced the geometric gate,
   because a size threshold cannot tell a far hand from a small piece of clutter.
5. **Agreement rules (dark build only).** In the final experimental build a lock needs
   three consecutive agreeing passes, and a tracking candidate that jumps off the hand
   does not move the cursor until it repeats.
6. **Plausibility gate, smoothing, cursor.** Positions that imply faster-than-possible
   hand movement are dropped, the accepted position is smoothed with a 0.35 s time
   constant, and the cursor follows at a capped speed so one wrong detection cannot fling
   the pointer across the screen.

## Layout

```
hand_mouse_cursor.py        the application (all the logic above)
hand_mouse_dark.py          entry point for the dark/experimental variant
config.py                   every tuned value, and which weights go with which result
crop_verifier.py            the CNN verifier (SmallCNN) and its inference wrapper
baseline_classifier.py      8-feature logistic regression I compared the CNN against

blazepalm/                  vendored BlazePalm PyTorch port (MIT, see Third-party below)
  blazepalm.py              the detector
  hand_pipeline.py          letterboxing, box decoding, rotation-normalised crop
  handlandmarks.py          landmark model, used offline as a labelling arbiter
  palmdetector.pth          original detector weights
  HandLandmarks.pth         landmark weights
  anchors.npy               2944 anchors for the detector

Training
  train_cnn_verifier.py     trains the verifier on the base dataset
  train_vll_verifier.py     same trainer, used for the bigger and dark-tuned models
  train_verifier.py         trains the logistic baseline
  blob_verifier.py          builds blob crops and trains the blob-pipeline verifier
  finetune_detector_dark.py fine-tunes the detector heads on the target condition

Data
  fetch_hagrid.py           small downloader for the HaGRID parquet mirror
  fetch_hagrid_bulk.py      resumable bulk downloader for the 512px HaGRID shards
  build_verifier_dataset.py builds the base crop set from the parquet mirror
  build_big_dataset.py      mines crops from the 512px shards, checks them with the arbiter
  filter_mined.py           the arbiter filter as a standalone step
  build_vll_dataset.py      builds the target-condition crop set, holds the augmentation
  build_dark_dataset.py     adds the dark and distance augmentation

Evaluation
  evaluate_pipelines.py     the main harness, reproduces the comparison tables
  eval_motion_gate.py       7-clip integrated comparison (hand frames vs clutter locks)
  eval_vll_pipeline.py      target-condition pipeline comparison
  eval_detector_models.py   detector-only localisation, original against fine-tuned
  gt_clips.py               caches MediaPipe truth for the 7 original clips
  gt_clips_static.py        the lag-free static-mode truth used as the yardstick
  gt_verylowlight.py        truth for the target clip
  diagnose_verylowlight.py  stage-by-stage funnel, says where each frame is lost
  diagnose_acquire_consensus.py  measures candidate positions to pick the agreement parameters
  diagnose_lock_events.py   counts wrong-place cursor events
  diagnose_blob_crops.py    compares blob crops against training-style crops
  sweep_*.py                threshold and sensitivity sweeps
  calibrate_motion_blobs.py, noise_margin_blobs.py, motion_probe.py   measurement tools
  arbiter_vll.py, critic_vll_cast.py   independent label and colour-cast checks

Tests
  test_startup.py           every start-up configuration, plus the locator loop
  test_acquire_consensus.py the agreement rules, including controls that must fail
  test_v10_equivalence.py   proves a late refactor did not change shipped behaviour
  test_moving_clutter.py    the blob path against a synthetic moving non-hand object
  test_cursor_controller.py unit tests for the cursor state machine

build_release.py            PyInstaller packaging with a start-up sanity check
VERSION_LOG.md              every build I made, including the ones that were broken
results/                    evaluation outputs for the configurations in the report
figures/                    the figures used in the report
docs/                       design and measurement notes, one per round
```

The eight recorded clips sit in the top level of the repo, because the evaluation
harness reads them straight from the working directory.

## Setup

Python 3.10. I used PyTorch on CPU only (no GPU), OpenCV, NumPy, and pyautogui for the
cursor movement. Everything else the scripts import is standard library.

```
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

`requirements.txt` lists the versions I actually ran. If you only want the evaluation
harness you do not need pyautogui or PyInstaller, but the app needs the first of those to
move the pointer at all.

The application runs from source:

```
python hand_mouse_cursor.py                     # default configuration
python hand_mouse_dark.py                       # experimental dark variant
python hand_mouse_cursor.py --no-mouse          # debug window, pointer untouched
```

It opens a window with a debug overlay, and you press `q` to quit. With no webcam
attached it prints `could not open webcam` and exits, which is what the automated tests
rely on.

## Data

### The recorded clips

I recorded eight clips myself, with my own hand, in my own room. They are in the repo
because the evaluation harness reads them directly, and I wanted the reported results to
be reproducible without my footage.

| clip | resolution | fps | frames | condition |
|---|---|---|---|---|
| CloseLightMoving.mp4 | 640x480 | 29.9 | 174 | hand close, daylight, moving |
| CloseLightStill.mp4 | 640x480 | 27.2 | 209 | hand close, daylight, held still |
| CloseDarkStill.mp4 | 640x480 | 15.0 | 94 | hand close, dark, held still |
| FarLight.mp4 | 1920x1080 | 30.0 | 195 | hand far, daylight |
| FarDark.mp4 | 1280x720 | 30.0 | 230 | hand far, dark |
| VeryLowLight.mp4 | 640x480 | 15.0 | 651 | hand close, artificial light, slightly dark room (my target condition) |
| NohandLight.mp4 | 640x480 | 15.1 | 45 | no hand, static room, daylight |
| NohandDark.mp4 | 640x480 | 15.1 | 53 | no hand, static room, dark |

If you record your own instead, keep the filenames and keep the two no-hand clips. The
harness assumes a static camera and an empty room for those two, and any accepted
position on them counts as a false lock by construction.

### Ground truth

`gt_clips.npz` and `gt_verylowlight.npz` hold cached MediaPipe output for those clips and
are already in the repo, so you do not need to regenerate them. The yardstick for
everything I report is MediaPipe's **static** mode (per frame, no tracker state, so no
lag), with gaps filled only within +-5 frames, applied identically to my pipeline and to
the MediaPipe baseline.

Regenerating them needs the `mediapipe` package in a separate environment:

```
python gt_clips.py                 # the 7 original clips
python gt_clips_static.py          # adds the lag-free static truth
python gt_verylowlight.py          # target clip
```

### HaGRID

I trained the verifier on crops mined from HaGRID. The dataset is not in this repo: the
subset I used is 10 GB, and the licence would not let me redistribute derived crops
anyway. Here is how to rebuild it.

```
# 1. pull about 10 GB of the 512px shards (resumable, safe to re-run)
python fetch_hagrid_bulk.py --gb 10

# 2. mine crops and check each one with the landmark arbiter
python build_big_dataset.py --max-hagrid 18000 --min-presence 0.5

# 3. optionally the smaller base set, if you want to retrain from scratch
python build_verifier_dataset.py

# 4. the target-condition crops, from my own clip plus the augmented copies
python build_vll_dataset.py
python build_dark_dataset.py
```

The shards carry `image` and `label` only, with no bounding boxes, which is why crops
have to be mined with the detector rather than cropped geometrically. Two things I
learned doing this, both of which matter for reproducing my numbers:

- Mining a positive as "the detector's top detection on an image labelled with a
  gesture" is wrong about **42% of the time**, because that conflates "this image
  contains a hand" with "this detection is the hand". `--min-presence 0.5` runs the
  landmark model's presence head over every mined crop and keeps only confirmed ones.
  Leaving that out gave me a verifier with AUC 0.8081 instead of 0.9466.
- Label 13 (`no_gesture`) is skipped. Those images show natural hand postures, so they
  are neither clean positives nor clutter.

The augmentation matches the photometric statistics I measured on the target clip (mean
BGR 32.9, 38.7, 57.4) and simulates distance by downscaling and upscaling. It is applied
to **both** classes, so brightness and sharpness cannot become shortcuts for the
classifier.

## Training

### The crop verifier

```
python train_cnn_verifier.py --dataset verifier_dataset --out verifier_cnn.pt --epochs 8
python train_vll_verifier.py --dataset big_dataset --out verifier_cnn_big.pt --epochs 8
python train_vll_verifier.py --dataset dark_dataset --out verifier_dark.pt --epochs 8
python blob_verifier.py
```

All of these use the same recipe: Adam at 2e-3, batch 64 (128 for the blob verifier),
8 epochs, binary cross entropy on the raw logit, and class balance handled by
subsampling the majority class rather than by weighting the loss. Each run prints the
held-out metrics for both the previous model and the new one on identical splits, which
is how I compared them.

### The detector fine-tune

This is the one place I touched BlazePalm. I fine-tuned the six 1x1 prediction heads and
left everything else frozen (43,966 of 1,763,358 parameters, 2.49%). The original
`palmdetector.pth` is untouched; the fine-tuned weights go to a separate file.

```
python finetune_detector_dark.py --unfreeze heads --epochs 20
```

Two things in that script are there because I got them wrong first:

- **Anchor matching.** My first attempt labelled every anchor whose centre fell inside
  the ground-truth box, about 96 of 2944 anchors per box. That diffuse signal taught the
  score head to fire everywhere: anchors above 0.5 went from 13 to 337, and localisation
  containment went to 0.0% on every clip. The shipped version keeps only the 6 anchors
  nearest the box centre.
- **A collapse guard.** Before saving, the script counts how many anchors score above 0.5
  on a light frame and refuses to save if there are more than 100. The original model
  measures about 13 and the shipped fine-tune measures 8.

### The logistic baseline

```
python train_verifier.py --dataset verifier_dataset --out verifier_lr.pt
```

I wrote this to check whether a CNN was justified at all. It is not a close comparison:
eight hand-designed features over a single linear layer reach AUC 0.7884 against the
CNN's 0.9650, and at a matched operating point it is worse than the plain size gate it
would have replaced (it loses 120 hand frames and adds 34 clutter locks).

## Evaluation

### The main command

This is the one that produces the comparison table in my report:

```
python evaluate_pipelines.py
```

It writes `build/pipeline_comparison.csv` and `build/pipeline_summary.txt`. Both
pipelines are scored against the same MediaPipe static-mode truth on the same frames. A
reported position counts as correct if it lands within `max(0.06 frame widths, 0.6 x palm
width)` of a truth hand.

It takes a couple of minutes on CPU for all eight clips, at about 20 ms per frame. For a
quick check that everything is wired up, add `--frames 20` to cap each clip.

### Which configuration produced which result

Every number in the report maps to one of these commands, and the weights for each are in
the repo.

| Report result | Weights and flags | Command |
|---|---|---|
| Main comparison table (MediaPipe against my pipeline, default configuration) | `verifier_cnn.pt` at threshold 0.30, original detector, blobs off | `python evaluate_pipelines.py` |
| The low-light A/B row of the same table | `verifier_cnn_big.pt` at threshold 0.61 | `python evaluate_pipelines.py --verifier verifier_cnn_big.pt --threshold 0.61` |
| The dark variant, three-way comparison | `palmdetector_dark.pth`, `verifier_cnn_big.pt` at 0.61, motion blobs on | live use: `python hand_mouse_dark.py`; for the table: `python evaluate_pipelines.py --detector palmdetector_dark.pth --verifier verifier_cnn_big.pt --threshold 0.61 --motion-blobs` |
| The motion-blob ablation row | as above with blobs off | `python evaluate_pipelines.py --detector palmdetector_dark.pth --verifier verifier_cnn_big.pt --threshold 0.61` |
| The agreement rules, final dark build | dark stack plus both rules | `python evaluate_pipelines.py --detector palmdetector_dark.pth --verifier verifier_cnn_big.pt --threshold 0.61 --motion-blobs --acquire-consensus 3 --acquire-consensus-tol 0.05 --track-consensus 3 --track-consensus-tol 0.06` |
| The agreement-rules ablation, acquisition only | dark stack, acquisition rule only | the same command without the two `--track-consensus` flags |
| Detector localisation, original against fine-tuned | both detectors, no verifier involved | `python eval_detector_models.py --models palmdetector.pth palmdetector_dark.pth` |
| The stage funnel for the target condition | default chain, per-stage counts | `python diagnose_verylowlight.py` |
| Wrong-place cursor events, before and after | dark stack, rules on and off | `python diagnose_lock_events.py` |
| The parameters for the agreement rules | measured candidate positions | `python diagnose_acquire_consensus.py` |
| Threshold sweeps | all verifier models | `python sweep_verifier_threshold.py` |

Outputs for the first six are already in `results/`, so you can read the numbers without
re-running anything.

### What the numbers came out as

Aggregate over the six hand clips, 1319 truth frames, identical yardstick:

| pipeline | accepted | precision | recall | F1 | no-hand false locks |
|---|---|---|---|---|---|
| MediaPipe (tracking) | 1129 (72.7%) | 88.3% | 75.6% | **81.5%** | 0.0% |
| mine, default | 589 (37.9%) | 81.2% | 36.2% | **50.1%** | 0.0% |
| mine, low-light A/B | 633 (40.8%) | 79.8% | 38.3% | **51.7%** | 0.0% |
| mine, dark variant | 850 (54.7%) | 63.4% | 40.9% | **49.7%** | 1.0% |
| mine, dark variant with the agreement rules | 602 (38.8%) | 74.1% | 33.8% | **46.4%** | 0.0% |

Per clip F1, for the two configurations that matter most:

| clip | MediaPipe | default | dark variant |
|---|---|---|---|
| CloseDarkStill | 99.5 | 93.8 | 12.1 |
| FarLight | 74.6 | 32.3 | 52.6 |
| FarDark | 77.8 | 14.9 | 51.7 |
| VeryLowLight | 65.1 | 8.8 | 31.8 |

Close range is within 6 to 14 points of MediaPipe. Distance and low light are not. The
specialised dark variant is a large improvement over my own default in the target
conditions, and much worse for close hands held still.

### Diagnostics and sweeps

```
python eval_motion_gate.py              # 7-clip hand frames vs clutter locks
python eval_vll_pipeline.py             # target clip, held-out blocks marked
python diagnose_blob_crops.py           # why the blob verifier behaves as it does
python sweep_motion_sensitivity.py      # blob pixel threshold and area floor
python sweep_blob_threshold.py
python calibrate_motion_blobs.py        # blob ROI geometry
python noise_margin_blobs.py            # how much sensor noise the gate survives
python arbiter_vll.py                   # independent hand-or-not check on detections
python critic_vll_cast.py               # how the verifier reacts to colour shifts
```

## Tests

```
python test_startup.py              # every start-up path, plus the locator loop
python test_acquire_consensus.py    # the agreement rules, with controls
python test_v10_equivalence.py      # refactor equivalence
python test_cursor_controller.py    # cursor state machine unit tests
python test_moving_clutter.py       # the blob path against a moving non-hand object
```

`test_startup.py` is the one that matters most. I shipped a build that crashed on
start-up because `blob_verifier` was only assigned inside an `if` branch, and every smoke
test I ran that round passed a flag that avoided the branch. The suite now starts the app
in every source configuration, runs the locator over 40 real frames headlessly (so a
machine with no webcam still exercises the code that crashed), and checks that a bare
model filename resolves from a different working directory. Its packaged-build section is
skipped automatically when there is no `dist/` directory.

`test_v10_equivalence.py` exists because I restructured the decision function late on and
needed to prove the shipped behaviour had not moved. It drives the current code and a
copy of the previous logic with the same random inputs and asserts that every state field
matches after each step.

## What is not in this repository

- **The packaged `.exe` builds.** Each is around 580 MB of compiled artefact. I kept 16
  versioned builds during the project and logged all of them in `VERSION_LOG.md`,
  including `HandCursor_v9`, which is broken and should not be used.

  Two of those builds are the ones worth having, and both rebuild in a few minutes:

  ```
  # the recommended general-purpose build (default configuration)
  python build_release.py --entry hand_mouse_cursor.py --name-base HandCursor ^
      --summary "general-purpose build"

  # the final experimental dark build (fine-tuned detector, motion blobs, both agreement rules)
  python build_release.py --entry hand_mouse_dark.py --name-base HandCursorDark ^
      --summary "dark-specialised variant"
  ```

  Each command packages the source into `dist/<name>/`, runs the resulting exe directly
  with cwd set to its own folder (the double-click path), checks that the verifier loads
  and that it reaches the webcam step, then appends a row to `VERSION_LOG.md`. It refuses
  to overwrite an existing build, so the version numbers keep incrementing.

  They are not in this repo because they cannot be. `_internal/torch/lib/torch_cpu.dll`
  alone is 291 MB, and GitHub rejects any push containing a file over 100 MB. Zipping a
  build only gets it to about 200 MB, which is still over the limit, so the choice was
  between Git LFS (which needs the grader to have git-lfs installed, and would use most
  of the free storage quota) and leaving them out. I left them out.
- **The HaGRID data and the crop datasets.** 10 GB of images plus about 1.4 million
  generated crops. The download and build commands are above.
- **Intermediate evaluation logs.** `results/` holds the outputs for the configurations
  in the report, not every run I made.
- **Exploration scripts that led nowhere.** An early OpenCV and MediaPipe version of the
  app, a landmark-based tracker I abandoned when I reset the scope, a one-euro-filter
  tuner, and a set of tests for modules that no longer exist. Leaving them in would have
  made the repo look like a working directory rather than a project.
- **My internal working notes** where they duplicate the README or the design docs.

## Third-party code and licences

- `blazepalm/` is the PyTorch port of Google's BlazePalm by Vidur Satija
  (https://github.com/vidursatija/BlazePalm). It is **MIT licensed** and I have kept the
  original `LICENSE` file in that folder. I vendored the four Python files and the
  weights, not the iOS app or the model converters.
- `palmdetector.pth`, `HandLandmarks.pth` and `anchors.npy` are ported from Google's
  MediaPipe models (`palm_detection.tflite`, `hand_landmark.tflite`). **MediaPipe is
  Apache 2.0**, which permits redistribution with attribution. The port converted the
  format; I have not modified these files.
- `palmdetector_dark.pth` is my fine-tune of the six heads of `palmdetector.pth`, so it
  carries the same Apache 2.0 terms as the original weights.
- The verifier models (`verifier_cnn.pt`, `verifier_cnn_big.pt`, `verifier_cnn_vll.pt`,
  `verifier_dark.pt`, `blob_verifier.pt`, `verifier_lr.pt`) are mine.
- `verifier_cnn_big_UNFILTERED.pt` is kept on purpose. It is the model I trained before I
  found the labelling error described above, and it is the evidence for that section. It
  is not used in any reported result.
- **HaGRID is not redistributed here.** The dataset is CC BY-SA 4.0 and I am not
  confident about the share-alike position on derived crops, so the images and crops stay
  out of the repo and the README explains how to rebuild them. If you rebuild them, the
  same terms apply to whatever you generate.

## Limitations

I would rather state these than have someone find them later.

- All my real dark data comes from **one person, one room, one lamp, one camera**, about
  200 labelled frames. The held-out dark frames are held out by frame, not by condition,
  so I can claim "this lighting", not "artificial light in general".
- Every model was trained **once**, with seed 0. I have no variance estimate for any of
  them. Re-running the same evaluation on the same model varies by about +-1.3 F1 points
  between sessions, because the harness passes wall-clock timings into the plausibility
  gate, so differences below that are not meaningful.
- MediaPipe is not only my baseline, it is also the source of my ground truth and the
  arbiter I used to clean the training labels. If its hand detection is weaker for some
  hands than others, that bias is in my data too.
- The motion-blob path locks onto a synthetic moving non-hand object every time it locks
  at all. No recorded clip contains real moving clutter, so that failure mode is untested
  on real footage.
- Nothing detects camera motion. The motion gate assumes a static camera.
- There is no click action and no gesture vocabulary. The system moves a pointer.
- The agreement rules cut wrong-place cursor movement by 59% and take the no-hand
  false-lock rate to zero, but they cost recall, and they make `CloseDarkStill` worse
  rather than better, because on that clip the true detections are the brief ones and the
  false ones are steady. Agreement over time cannot fix a stable false positive.

The design notes in `docs/` go into each round in more detail, including the things that
did not work.
