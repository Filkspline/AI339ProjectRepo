# Build version log

Versioned double-click builds. **Never overwritten** - older versions stay in
`dist/` for rollback/comparison. Sizes are MiB.

Each program keeps its **own** version sequence. Rebuild with:

```
:: the tracking app
.venv-blazepalm\Scripts\python.exe build_release.py --summary "vN - what changed"

:: the cursor app
.venv-blazepalm\Scripts\python.exe build_release.py --entry hand_mouse_cursor.py ^
    --name-base HandCursor --summary "vN - what changed"
```

Either command auto-increments that program's version, builds **onedir** into
`dist/<name>/`, refuses to overwrite an existing build, runs the exe directly
(with no venv active) and confirms it reaches the webcam-open step without a
traceback, then appends a row to the table below. Add `--onefile` only for a
convenience single-file build.

## Pre-versioning artifacts (kept, superseded)

Built before versioning was introduced; left in place for reference.

| artifact | mode | size | note |
|---|---|---|---|
| `dist\HandTrackerPyTorchDir\` | onedir | 556 MB | same code state as `HandTrackerPyTorch_v1` |
| `dist\HandTrackerPyTorch.exe` | onefile | 189 MB | onefile variant; extraction could not be verified in this sandbox |
| `dist\HandMouseMediaPipe\` | onedir | 604 MB | the original MediaPipe `hand_mouse.py`, not the PyTorch port |
| `dist\HandMouseMediaPipe.exe` | onefile | 226 MB | onefile variant of the above |

## Versioned builds

| version | date | mode | size | summary | sanity |
|---|---|---|---|---|---|
| `HandTrackerPyTorch_v1` | 2026-09-13 | onedir | 556 MB | baseline: box-size FP gate (0.15W) + consecutive-detection gate + motion prediction + One Euro smoothing (min_cutoff=1.0, beta=0.2) | OK (reached webcam step) |
| `HandCursor_v1` | 2026-09-13 | onedir | 577 MB | new app: acquisition dwell + rate-limited cursor + hold-on-loss state machine (dwell 1.0s, speed cap 1.0 screen-width/s, loss timeout 3.0s, lock radius 0.12; box-size FP gate retained, no landmarks/gate/predictor/smoothing) | OK (reached webcam step) |
| `HandCursor_v2` | 2026-09-13 | onedir | 577 MB | search-window crop-and-zoom tracking (REACQUIRE/TRACK modes, window snaps back to 3x box on hit, widens 1.4x/miss, 3s loss timeout), plausibility gate (3 frame-widths/s), heavy EMA smoothing (tau 0.35s), lock needs 2 consecutive hits; in-window size gate relaxed to 0.04 | OK (reached webcam step) |
| `HandCursor_v3` | 2026-09-13 | onedir | 657 MB | crop verifier replaces the relaxed in-window size gate (SmallCNN 50k params, verifier_cnn.pt, threshold 0.30); measured vs MediaPipe GT on the clips: 486/499 true hand frames kept, clutter locks 156 -> 32; bundle now ships verifier_cnn.pt | OK (reached webcam step) |
| `HandCursor_v4` | 2026-09-16 | onedir | 578 MB | acquisition dwell/hover-square removed: a verifier-accepted position locks and drives the cursor immediately (speed cap + heavy smoothing unchanged); pyarrow excluded from the bundle | OK (reached webcam step) |
| `HandCursor_v5` | 2026-09-16 | onedir | 578 MB | acquisition motion gate: frame differencing on a 320px-wide 5-frame ring, changed-pixel fraction >= max(0.01, 3x median background tile), REACQUIRE-only and ANDed with the size gate, dropped in TRACK; frame-wide gain compensation + global-change suspension; `--no-motion-gate` rollback | OK (reached webcam step) |
| `HandCursor_v6` | 2026-09-19 | onedir | 578 MB | REACQUIRE size gate relaxed 0.15 -> 0.10, tied to the motion gate that now carries the clutter rejection (`--no-motion-gate` keeps 0.15, so it stays a true v4 rollback); restores on-hand far-range acquisition. Integrated GT eval, same frames/matching as the v3/v4 baseline: **469 hand frames / 15 clutter locks** (v3/v4 486/32, v5 438/3), FarLight 28/6 and FarDark 22/8 (baseline 25/26 and 6/3), both no-hand clips 0 acquisitions | OK (reached webcam step) |
| `HandCursor_v7` | 2026-09-19 | onedir | 578 MB | bundles the dark/artificial-light-tuned verifier `verifier_cnn_vll.pt` for live A/B via `--verifier verifier_cnn_vll.pt`; **default verifier unchanged** because the retrained model regresses the daylight clips (clutter locks 15 -> 37); no pipeline-code change. See `verylowlight_diagnosis.md` | OK (reached webcam step) |
| `HandCursor_v8` | 2026-09-19 | onedir | 578 MB | motion-blob candidate proposals behind `--motion-blobs`, **OFF by default**: connected components on the existing motion-difference ring propose regions outside the detector's box and the verifier confirms them (8-way rotation search, 0.10 prefer-margin in TRACK). Proposal lands on the hand in 42.2% of hand frames vs the detector's 35.7%, but the verifier accepts 88-94% of OFF-hand blobs, so on VeryLowLight clutter locks go 98 -> 405 and a synthetic moving non-hand object is locked 7/7. Identical numbers to v6 on the 7 clips. See `motion_blobs.md` | OK (reached webcam step) |

**Where to start if you are picking this up:** the comparison numbers are reproducible
with the commands in the README, and the outputs I kept are in `results/`.

* Recommended general use: `HandCursor_v10` with no flags.
* Recommended for dim / artificial light:
  `HandCursor_v10.exe --verifier verifier_cnn_big.pt --verifier-threshold 0.61`
  (the threshold must be passed with it; at the default 0.30 that model is far too
  permissive).
* `dist\HandCursor_v9` is broken - do not use. `test_startup.py` now guards this
  class of failure permanently.

Notes on the table:

* `HandCursor_v3` is 80 MB larger than v2 for no functional reason: torch's
  PyInstaller hook pulled in **pyarrow** (79 MB), which is only used by
  `build_verifier_dataset.py` and never at runtime. `HandCursor_v4` passes
  `--exclude-module pyarrow` and is back to 578 MB. v3 is left exactly as built
  (no version is ever overwritten).
* `HandCursor_v3` and later load the verifier from
  `_internal/models/verifier_cnn.pt` and print
  `TRACK gate = verifier p_hand >= 0.30` on startup; `--no-verifier` reverts it
  to the v2 size gate at runtime.
* `HandCursor_v4` drops the dwell/hover-to-lock mechanic entirely.
* `HandCursor_v5` adds the acquisition motion gate and prints
  `ACQUIRE motion gate = ...` on startup; `--no-motion-gate` restores pre-v5
  behaviour.
* `HandCursor_v6` relaxes the acquisition size gate and prints
  `ACQUIRE size gate = box width >= 0.1 x frame`. The relaxation is only
  justified by the motion gate, so the two are tied in code: with
  `--no-motion-gate` the size gate returns to 0.15 and the build behaves like v4.
* Measured cost of v5/v6: a hand that is *held still* takes ~30
  frames (~1 s) to acquire, because the motion gate must see micro-tremor plus
  the 5-frame ring fill. CloseLightStill therefore tracks 180 frames in v6
  against the baseline's 209.
* Full per-clip numbers and the config comparison are in `motion_gate.md`;
  re-measure with `eval_motion_gate.py` (add `--match-r 0.12` to reproduce the
  baseline's matching).
* `HandCursor_v7` adds no pipeline code: it ships `verifier_cnn_vll.pt`, the model
  retrained for the dark/artificial-light condition, alongside the default. Switch
  models at runtime with `--verifier verifier_cnn_vll.pt` (the exe resolves it from
  `_internal/models/`). The default was left alone because the retrained model
  costs clutter locks on the already-validated daylight clips (15 -> 37 overall,
  FarLight 6 -> 29) and only moved the VeryLowLight held-out pipeline result from
  7/200 to 8/200 frames - see `verylowlight_diagnosis.md` for the full diagnosis.
* `HandCursor_v8` adds the motion-blob candidate path behind `--motion-blobs`
  (**off by default**). With the flag off the code path is identical to v6/v7
  (tests still 50/50, same 469/15 on the 7 clips). With it on, the app prints a
  `WARNING: measured in the target lighting, the verifier accepts 88-94% of
  OFF-hand motion blobs`. Recommendation and full numbers: `motion_blobs.md`.
| `HandCursor_v9` | 2026-10-04 | onedir | 578 MB | **BROKEN, DO NOT USE.** Attempted build of the motion-blob-verifier round. It FAILED its sanity check: `blob_verifier` was only assigned inside `if args.motion_blobs:`, so the app raised UnboundLocalError at startup in its DEFAULT configuration, which is the path a double-click takes. The artifact was still written to `dist/HandCursor_v9/`; it crashes immediately. Superseded by `HandCursor_v10` | **FAILED** |
| `HandCursor_v10` | 2026-10-04 | onedir | 578 MB | v10: FIXES a startup crash present since v9 (blob_verifier was unbound unless --motion-blobs was passed, so the default double-click path died immediately) + bundles verifier_cnn_big.pt for live A/B; default verifier unchanged because at matched precision it triples clutter locks on the validated clips (15 -> 45, FarLight 6 -> 37) | OK (reached webcam step) |

* `HandCursor_v9` is the one broken artifact in `dist/`: the build completed and
  wrote its files, but the startup sanity check failed, so it is logged FAILED.
  Cause: `blob_verifier` was referenced outside the `if args.motion_blobs:` block
  that assigned it, i.e. the **default** configuration - the one a double-click
  runs - died with `UnboundLocalError` before reaching the webcam. It went
  unnoticed because every local smoke test of that round used `--motion-blobs`, and
  the build whose sanity check would have caught it was interrupted before the
  check ran. `test_startup.py` now launches the app as a subprocess in all seven
  configurations and asserts each reaches the webcam step, so this class of bug
  cannot recur silently. `HandCursor_v10` is the first working build since v8.
* `HandCursor_v10` fixes that and bundles `verifier_cnn_big.pt`, the 130k-crop /
  real-HaGRID-volume verifier, for live A/B:
  `HandCursor_v10.exe --verifier verifier_cnn_big.pt --verifier-threshold 0.61`.
  The default is unchanged: at matched precision the new model triples clutter
  locks on the validated clips (15 -> 45, FarLight 6 -> 37) while improving
  VeryLowLight F1 (8.8 -> 16.5). Full numbers in `verifier_v2_hagrid_volume.md`.
| `HandCursorDark_v1` | 2026-10-04 | onedir | 585 MB | **EXPERIMENTAL, separate from v10.** Dark-specialised variant: BlazePalm's six 1x1 heads fine-tuned (2.49% of params; +8.0 points VeryLowLight holdout localisation at +0.4 mean on the 7 unseen light clips; overfit gap 4.2 -> 1.5) + `verifier_cnn_big.pt` @0.61 + motion blobs ON. Three-way: VeryLowLight F1 8.8 -> 31.8, FarDark 14.9 -> 51.7, FarLight 32.3 -> 52.6, but CloseDarkStill 93.8 -> 12.1 and **NohandLight false locks 0.0% -> 2.2% (26.7% with blobs off) - the first non-zero no-hand rate in the project**. See `dark_variant.md`. **THIS BUILD IS BROKEN, DO NOT USE: it crashed about a second after the camera opened (verifier resolved relative to cwd only, so it went inactive on a double-click and the blob path then called a None verifier). Its build-time sanity check said OK because it ran from the wrong directory and on a machine with no camera - see the v2 entry below. Superseded by `HandCursorDark_v2`** | **FAILED IN THE FIELD** |

* `HandCursorDark_v1` is a **new, separate naming sequence** (`--name-base
  HandCursorDark`) and is explicitly experimental. It is the only variant allowed
  to trade light-condition performance away, and it does: run it for far / dim
  conditions with a moving hand. **It has a non-zero false-lock rate on
  NohandLight (2.2%, and 26.7% with motion blobs off), caused by the fine-tuned
  detector** - the first such regression in the project's history, and the reason
  it is not the default. `HandCursor_v10` remains the recommended build for
  general use, with its own 0.0% no-hand rate intact.
* The fine-tuned detector is `palmdetector_dark.pth`; **`palmdetector.pth` is
  untouched and byte-identical**, so v10 and every earlier build behave exactly as
  before. The app gained one backward-compatible flag, `--detector`, to load
  alternative weights; `test_startup.py` still passes 7/7 configurations.
* Full round write-up, including the failed first fine-tune attempt and the
  collapse guard added because of it: `dark_variant.md`.

### HandCursorDark_v2 - crash fix for the v1 defect

`HandCursorDark_v1.exe` crashed roughly a second after the camera opened. Cause:
the dark build requests `--verifier verifier_cnn_big.pt` by bare name, and model
lookup was relative to the process **cwd only**, so on a double-click (cwd = the
exe's folder) the verifier silently went inactive. Motion blobs are ON by default
in this variant, so the blob path then passed a `None` verifier into rotation
scoring: `TypeError: 'NoneType' object is not callable`, around frame 5.

Fixed in the app: one `resolve_model_path()` for every model flag (as-given, then
bundled `models`, then script dir); an explicitly requested model that cannot be
found is now fatal instead of degrading silently; the blob path no longer assumes a
verifier exists; `--motion-blobs` is refused when no verifier is available; new
`--no-motion-blobs` flag.

Fixed in the checks, which is the part that matters - the defect passed every
previous check:

* `test_startup.py` now runs 4 sections (20 checks): every source flag combination
  including the **zero-argument double-click path**; the locator loop **headlessly
  over 40 real frames**, including the exact crash configuration (blobs on,
  verifier None), so no camera is needed to exercise the code that crashed; bare
  model names resolved from a **foreign working directory**; and every frozen build
  launched with **zero arguments and cwd = the exe's own directory**.
  `HandCursorDark_v1` is retained as a **canary** - the suite asserts it still
  reports "verifier NOT loaded", so the test fails if it ever stops detecting the
  known-bad build.
* `build_release.py`'s sanity check now launches the exe with **cwd = its own
  directory** (a double-click), requires the verifier to load for
  `hand_mouse_dark.py` as well as `hand_mouse_cursor.py` (the dark entry was
  previously exempt, so the build could pass with no verifier at all), and says
  so when there is no webcam so "reached the webcam step" is never read as
  "the loop works".

`HandCursorDark_v1` is left in `dist/` untouched. The measured dark-variant numbers
in the table above still apply - the detector, verifier and blob logic are
unchanged; only model *resolution* was broken.

| version | date | mode | size | summary | sanity |
|---|---|---|---|---|---|
| `HandCursorDark_v2` | 2026-10-05 | onedir | 585 MB | Crash fix for v1: cwd-independent model resolution for `--detector`/`--verifier`/`--blob-verifier` (via `resolve_model_path`), missing explicit model is fatal, no-verifier guard on the blob path, new `--no-motion-blobs`; build-time sanity check now uses cwd = exe dir and requires the verifier for both hand-mouse entries | OK (verifier loaded, reached webcam step, zero args, cwd = exe dir) |
### HandCursorDark_v3 - multi-pass spatial consensus on REACQUIRE

Requested after live testing: in slightly dark conditions the cursor is jumpy and
locks onto things like an armpit before self-correcting within milliseconds. The
change makes acquisition stop trusting one frame: a lock is committed only after
`--acquire-consensus 3` consecutive accepted passes that also **agree spatially**
(each within `--acquire-consensus-tol 0.05` frame widths of the run's mean). A
miss breaks the run; a disagreement restarts it with the disagreeing candidate, so
an alternating hand/other flicker never reaches N. TRACK is untouched, and N = 0
(the default, and everything v10 does) is exactly the old behaviour.

Both parameters come from `diagnose_acquire_consensus.py`, which replays the real
dark acquisition path with the locator pinned in REACQUIRE: genuine hand movement
between accepted frames is 0.046-0.057 frame widths at p95 on the moving clips
while the hand/other excursions this targets start at 0.069-0.074, so 0.05 sits
between them; N = 4 collapsed first-lock acquisition on VeryLowLight to 160 frames
(10.7 s) for about two further points, so N = 3. Full reasoning in
`consensus.md`.

Verification: `test_acquire_consensus.py` (11 checks) proves a 30-frame alternating
flicker never commits and a sustained location commits after exactly 3 passes --
including a control showing that with the consensus OFF the same flicker *does*
lock, and an integration pass over real frames where identical hits lock without
the consensus and do not with it. End-to-end across all clips: CloseDarkStill F1
10.8 -> 16.0, CloseLightStill 55.4 -> 55.3, FarLight 51.8 -> 52.6, FarDark
51.8 -> 51.8, VeryLowLight 31.7 -> 30.6, aggregate 49.7 -> 49.8, no-hand false
locks unchanged (2.2 % / 0.0 %).

**Measured limitation: this does not fix the live jumpiness.** Of
317 off-hand accepted frames across the hand clips, 307 are produced by TRACK and
only 10 by REACQUIRE (excursions 85 -> 82). The request scoped the change to
REACQUIRE, so it can only address about 3 % of them; most excursions are transient
(median 1-4 frames), which is the same signature as the reported symptom, so the
same agreement idea applied to the in-window TRACK candidate is the change that
would actually address it.

`HandCursor_v10` is unchanged and remains the default: the flag is off unless a
wrapper passes it, and only `hand_mouse_dark.py` does.

### HandCursorDark_v4 - "if it is not certain, it should not move the mouse there"

v3 fixed the *lock* but not the *cursor*, and the measurement of v3 said why: two
different unconfirmed candidates were moving the mouse.

1. **REACQUIRE was publishing pass 1.** `update()` returned a candidate as soon as it
   cleared the gates, whether or not the consensus had committed, so the cursor was
   dragged to the first pass while only the tracking window waited. Now `_on_hit`
   reports whether a position may be published, and an unconfirmed run publishes
   nothing at all (`CursorController` holds the cursor exactly still with no target).
2. **NEW `--track-consensus N`**: in TRACK, an in-window candidate further than
   `--track-consensus-tol 0.06` frame widths from the committed position does not
   move the cursor. It is held until N = 3 consecutive candidates agree on the new
   place, **or 6 frames pass, whichever comes first**. The bound exists because a
   hand moving steadily faster than the tolerance produces claims that never agree
   with each other: an unbounded hold would freeze the cursor until the 3 s loss
   timeout, turning a wrong jump into a dead cursor. So the rule delays a jump
   instead of vetoing it, and suppresses transient excursions only.

Measured (`diagnose_lock_events.py`, both configs in one process): off-hand cursor
placements 362 -> 164 (-55 %), **wrong-location excursions 95 -> 39 (-59 %)**,
persistent wrong locks (>= 10 frames) **9 -> 1**, accepted frames 912 -> 607 (-33 %).
End-to-end: precision 63.3 -> 74.1, recall 41.0 -> 33.8, F1 49.8 -> 46.4, and
**NohandLight false locks 2.2 % -> 0.0 %** - the first clean run on that clip for
this variant. With `--track-consensus-tol 0.09` instead: precision 70.2, F1 47.6.

**Regression, not hidden: CloseDarkStill F1 16.0 -> 6.2.** On that clip the hold
suppressed the correct detections more than the incorrect ones (on-hand TRACK frames
8 -> 3, off-hand 49 -> 33), because there the true detections are the *transient*
ones and the false ones are *stable*. A temporal-agreement rule preferentially
discards the truth in exactly that situation, which is direct evidence that this clip
needs the verifier to reject its steady false positives rather than more timing.
Very-low-light also pays: acceptance 30.9 % -> 17.5 %, i.e. the cursor is present for
less of the time. Escape hatches, one flag each: `--track-consensus-tol 0.09`,
`--track-consensus 0` (v3 behaviour).

Verification: `test_acquire_consensus.py` extended to 18 checks, including that
nothing is published before a lock is confirmed, that a 1-frame jump never moves the
cursor, that sustained relocation *is* followed, that within-tolerance motion is
never delayed, that a fast sustained move is picked up within the hold limit (the
freeze failure mode above), and controls showing the cursor *does* move in each case
with the hold off. v10's flags-off path is proven unchanged by
`test_v10_equivalence.py`: the pre-refactor state machine re-implemented verbatim and
driven against the real one over 13 random sequences (~6 300 steps), asserting all
seven state fields match after every step and every accepted hit was published.

| version | date | mode | size | summary | sanity |
|---|---|---|---|---|---|
| `HandCursorDark_v3` | 2026-10-05 | onedir | 585 MB | Multi-pass spatial consensus on REACQUIRE (3 consecutive accepted passes within 0.05 frame widths of the run mean): a flickering detector cannot commit a lock. Chosen from measurement; TRACK untouched. CloseDarkStill F1 10.8 -> 16.0 at no measurable cost on the far/dark clips, no-hand false locks unchanged. Does NOT fix the live armpit-jumping - 307 of 317 off-hand frames come from TRACK, which was out of scope | OK (verifier loaded, consensus ON reported, zero args, cwd = exe dir) |
| `HandCursorDark_v4` | 2026-10-05 | onedir | 585 MB | Never move the cursor to an unconfirmed position: REACQUIRE publishes nothing until its run is confirmed (previously the lock waited but the cursor was dragged to pass 1), and the new `--track-consensus 3 @ 0.06` holds an in-window jump until 3 candidates agree on the new place or 6 frames pass (bounded so a fast hand cannot freeze the cursor). Measured: off-hand placements 362 -> 164, excursions 95 -> 39 (-59 %), persistent wrong locks 9 -> 1, NohandLight false locks 2.2 % -> 0.0 %, precision 63.3 -> 74.1. Cost: recall 41.0 -> 33.8, F1 49.8 -> 46.4, CloseDarkStill regresses 16.0 -> 6.2 (its true detections are the transient ones). `--track-consensus-tol 0.09` or `--track-consensus 0` are the escape hatches | OK (verifier loaded, both consensuses ON reported, zero args, cwd = exe dir) |
