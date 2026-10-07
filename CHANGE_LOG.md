# Change log for the AIML339 final project revision

Every change made in this revision, what it did to the numbers, and why. Report numbers
quoted here are the ones in the previous draft unless stated otherwise.

## 1. Evaluation timing is now deterministic

**What changed.** `evaluate_pipelines.py`, `diagnose_lock_events.py`,
`diagnose_acquire_consensus.py`, `eval_motion_gate.py` and `eval_vll_pipeline.py` passed
wall-clock `dt` into the locator. The plausibility gate's speed limit (3 frame widths per
second) and the tracking miss timer are both time based, so replaying a clip faster or
slower than the machine happened to run changed which detections were accepted.
Timestamps now come from the frame index and the clip's own frame rate
(`clip_timestep()`), and nothing in the decision path reads the clock.

**Verification.** The same configuration was run three times. All 16 clip and pipeline
rows are identical across the three runs, except the `ms_per_frame` column, which
measures wall clock by design and is not a decision input.

**Consequence for the text.** The +-1.3 F1 run-to-run noise floor described in the
previous draft was a property of the harness, not of the models. It no longer applies to
evaluation, and the paragraph about it should be replaced by the determinism result.

**Numbers that moved because of this** (same weights, same clips, same truth, only the
timing changed):

| system | metric | before | after |
|---|---|---|---|
| Ours default | accepted frames, hand clips | 589 | 614 |
| Ours default | precision | 81.2% | 79.2% |
| Ours default | recall | 36.2% | 36.8% |
| Ours default | aggregate F1 | 50.1% | 50.3% |
| Dark v4 | accepted frames, hand clips | 602 | 581 |
| Dark v4 | aggregate F1 | 46.4% | 45.4% |

The MediaPipe baseline column does not use the locator and is unchanged by the timing
fix.

## 2. All systems evaluated in one session

**What changed.** The previous draft's dark columns came from a different session than
the light columns. `evaluate_all_systems.py` now evaluates MediaPipe, the default, the
big-verifier configuration, Dark v3 and Dark v4 in a single process, on the same footage,
against the same truth.

**Consequence.** The columns are directly comparable for the first time. This also revealed
that MediaPipe's own tracking output varies slightly between runs: a fresh run on the raw
frames gives 1131 accepted frames against the 1129 in the cached ground-truth run, about
0.1 F1 points. The whole table now comes from one run so this cannot flatter any system.

## 3. Wall-clock was also replaced in the two diagnostics

`diagnose_lock_events.py` and `diagnose_acquire_consensus.py` are used to justify the
agreement parameters, so they now use the same deterministic timing. `eval_motion_gate.py`
previously assumed 30 fps for every clip, which is wrong for the four clips recorded at
15 fps; it now reads each clip's frame rate.

## 4. CLAHE and gamma baselines added, and both lose to plain MediaPipe

**What changed.** Two preprocessing conditions were added in front of unmodified
MediaPipe Hands: CLAHE on the L channel in LAB (clipLimit 2.0, 8x8 tiles) and a luminance
gamma correction (gamma 0.5). Parameters are the usual textbook values, fixed in advance,
so nothing was tuned on any part of the data.

**Result, and it is negative.** On VeryLowLight, the fraction of frames with a detected
hand falls from 60.8% (raw) to 58.1% (CLAHE) and 59.6% (gamma). Aggregate F1 across the
hand clips falls from 81.6% to 80.4% (CLAHE) and 79.8% (gamma). FarLight loses the most:
82 accepted frames raw, 108 with CLAHE, and F1 75.0%, 76.7%, 66.7%.

**Consequence for the text.** The previous draft's future-work list suggested improving
detection in low light as the highest-return item. That remains the right conclusion, but
the specific intervention of classical contrast enhancement is now tested and does not
work, so the recommendation should shift to retraining the detector rather than to
preprocessing the input.

## 5. Clean train / validation / test split for VeryLowLight

**What changed.** The old scheme held out every third 50-frame block and used those
blocks for reporting. Those same blocks had also been used to choose the verifier
thresholds, the augmentation recipe and the fine-tune length, so they were not a test set.
A seeded three-way deal now assigns blocks to train, validation and test
(`vll_split.py`, split seed 0, recorded in `results/vll_split.json`):

    train  blocks 0, 2, 3, 4, 5      250 frames
    val    blocks 7, 10, 11, 13      151 frames
    test   blocks 1, 6, 8, 9, 12     250 frames

All selection uses train and validation only. The test blocks are evaluated once.

**Consequence.** Anything trained or tuned on the old held-out blocks is retrained behind
the new split: the verifier variants and the detector fine-tune both select their
checkpoints on validation blocks now. Numbers for VeryLowLight that came from the old
held-out set are not comparable with the new test-block numbers and are replaced.

**A finding that matters for how the results should be read.** The validation blocks are
not representative of the test blocks. For Dark v3, per-frame F1 is 14.6 on validation
blocks and 43.5 on test blocks. The clip's brightness drifts through the recording, so
the blocks are not exchangeable. That is a limitation of a single 651-frame clip, and it
is reported rather than smoothed over.

## 6. Human labels break the circular ground truth

**What changed.** Every number in the previous draft was scored against MediaPipe's own
static detector, which is also the baseline model family. `label_frames.py` selects 150
frames from the test blocks (30 evenly spaced frames from each of the five test blocks,
spanning frames 51 to 650) and records hand boxes or no-hand from a human, with the
MediaPipe box drawn on screen so the labeller can disagree with it.

**Status.** The tool, the frame list and a template CSV are written. The labels themselves
have to come from the project author; no human-label column can be reported until then.
What is already visible: MediaPipe detects a hand in only 61 of the 150 selected frames,
so the labels will bear directly on whether the other 89 are genuine no-hand frames or
MediaPipe misses. **Superseded by section 13: the labels were not collected, and the report
says so.**

## 7. Multi-seed training

**What changed.** The four verifier variants (11k, 130k, 32k target-condition, 84k
dark-augmented) and the detector fine-tune are trained with ten shared seeds. The
checkpoint is the epoch with the lowest validation loss; the operating threshold is a
per-variant hyperparameter chosen once on the validation blocks, so a seed changes the
weights only and the seed variance is not inflated by threshold selection. Training time,
parameter count, selected epoch, validation score and the collapse-guard result are logged
per run in `results/verifier_seeds/` and `results/detector_seeds/`.

**Consequence.** Every single-seed number in the previous draft should be quoted with the
mean and standard deviation from the ten seeds instead, and the section that said no
significance tests were run must be replaced.

## 8. Statistical tests

**What changed.** Paired per-frame McNemar tests plus 50-frame block bootstrap confidence
intervals for F1 differences, and paired across-seed tests (Shapiro-Wilk to choose between
a paired t-test and Wilcoxon signed-rank) with Holm-Bonferroni adjustment at alpha 0.05.
Full tables in `results/stats_paired_frames.csv` and `results/stats_across_seeds.csv`.

**Consequence.** Claims are now limited to what the tests support. Where a difference is
not significant, the report says the evidence is insufficient rather than implying a win.

## 9. Wording and claims

* The phrase "more than triples" and similar is removed unless the clean test blocks
  support it.
* Any number previously described as held out but produced from the old held-out blocks is
  either replaced with a test-block number or labelled as validation.
* The claim that the port is bit-accurate against the .tflite is removed unless the check
  is actually run; it is not run here.
* Added: the CLAHE and gamma references, and the HaGRIDv2 reference.
* Kept from the eariler draft because the new evidence still supports them: the
  CloseDarkStill explanation (the true detections are transient and the false ones stable,
  so temporal agreement discards the truth), and the note that the development machine has
  no webcam so every automated check runs headless.

## 10. Numbers re-measured under the fixed timing, and one that did not survive

**Re-measured.** `diagnose_lock_events.py` was re-run with both configurations in one
process (dark config, hand clips). The consensus result changes:

| | v3 (acquire only) | v4 (+ TRACK hold) |
|---|---|---|
| accepted frames | 844 | 581 |
| off-hand placements | 316 | 150 |
| **excursions** | **79** | **30** (−62 %) |
| persistent wrong locks (≥10 frames) | 7 | 1 |

The previous draft quoted 95 → 39 (−59 %) from a run made before the timing fix, so the
sentence "cut wrong-place cursor movement by 59 %" was replaced by the numbers above,
including the recall cost (40.0 % → 32.7 %) read from the same session's Table 1.

**Did not survive.** The draft said motion blobs "propose the hand slightly more often than
the detector's box contains it (42.2 % against 35.7 % of hand frames)". Re-running the
funnel in the current harness gives **36.8 %** against 35.7 % (same 291 static-truth hand
frames, same blob verifier at 0.80). The 42.2 % was a stale number from an earlier round
and is replaced. The rest of that paragraph re-measured exactly: 21.4 % proposed and 7.5 %
confirmed on detector-failure frames, about 14 frames or 4.8 points.

## 11. Attribution re-verified

The detector attribution in the report (291 static-truth hand frames, 187 lost at the
detector = 64.3 %, of which 77 with no detection and 110 off the hand) reproduced exactly
when the funnel was re-run in the current harness, so those numbers stand. Two additions:

* The same funnel run against the evaluation's own tracking-mode truth loses 310 of 396
  hand frames at the detector (78.3 %), so the attribution does not depend on which truth
  definition is used. One sentence added.
* The verifier's median probability on a hand-containing detection rises from 0.128 to
  0.995 between the two verifiers, which is the mechanism behind 5.5 % → 25.1 % survival.

## 12. Tooling corrections found while checking the tables

* `stats_tests.py` computed F1 as tp = correct ∧ truth with an accepted-but-off-hand frame
  counted as a false **negative**; `evaluate_all_systems.py` counts it as a false
  **positive**. Two different F1 values for the same system would have appeared in one
  report, so the statistics now use the same convention as Table 1 (verified: mp_raw
  0.81551, ours_default 0.50285, matching the table).
* The same script now reports two explicit scopes, `hand` (the six hand clips, the scope the
  report quotes) and `all` (which adds the no-hand clips). They give identical results here
  because no system accepts anything on a no-hand clip, and the CSV records which is which.
* `diagnose_verylowlight.py` printed `hmc.VERIFIER_THRESHOLD` (0.3) in its stage-funnel
  labels while the gate used the `--threshold` argument, so the big-verifier run was
  labelled "(>= 0.3)" in the log while running at 0.61, and the "% below threshold" line was
  computed at 0.3 as well. Labels and the computation now use the run's own threshold. The
  printed funnel numbers were always computed with the right threshold.
* The same file printed its blob and colour-cast sections twice (a duplicated block). The
  duplicate is removed; no number changed.
* `seeds_pipeline_eval.py --phase select` gave the dark pipeline the big verifier's
  threshold, which had been selected against the *original* detector. The dark pipeline runs
  a fine-tuned detector, so its threshold is now selected against that detector on the same
  validation blocks and recorded, with the detector and verifier named in `thresholds.json`.
* `report_tables.py`, `clip_table.py`, `make_latex_tables.py` and `gt_selfconsistency.py`
  were added so every table and quoted statistic can be printed from the result files
  instead of transcribed (`results/report_tables.txt`, `results/latex_fragments.tex`).

## 13. Human labels: not collected

The project author chose not to label the 150 frames. The report therefore does not show a
human-scored column; V-E states plainly that no labels were collected and that no claim is
validated by an annotator, keeps the tool and frame list as a prepared artifact, and adds
the one thing that can be measured without a human: MediaPipe's own two modes agree on
hand presence in 134 of those 150 frames (89.3 %), and on the 59 frames where both see a
hand the two positions are further apart than the evaluation's own match radius in 6
(10.2 %). Measured by `gt_selfconsistency.py`, written to
`results/human_labels/gt_selfconsistency.txt`.

## 15. A variant was being retrained on a third of its data

**What was wrong.** `train_verifier_seeds.py` derives each crop's split role from its
filename-derived source. The source token list did not include the VLL augmentation
suffix, so all 19,956 rows with source `hagrid_train_vll_aug` mapped to no role and were
dropped: the variant the report calls "32k VLL" was about to be retrained on 11,840 of its
32,027 rows, having silently lost the HaGRID training images put through the target
condition's augmentation recipe, which are the point of that variant. The original
`train_vll_verifier.py` used the whole manifest, so this was a regression introduced by the
multi-seed rewrite, not a property of the data.

**Fix.** `_vll_aug` added to the stripped tokens, and the variant retrained on the full
pool (all 32,027 rows now map to a role; about 27k are trainable). The other three variants
are unaffected: their row counts kept equal their manifest row counts (11,203, 130,532) or
their sources map cleanly (84,670 for the dark variant). Verified by comparing "rows kept"
against the manifest line counts, and recorded in `build/verifier_vll_refit.log`.

## 16. Arbiter ablation re-measured on one held-out pool

The report used to say "held-out AUC 0.8081 against 0.9650 for the small clean set", two
numbers from the earlier two-way split. Both models still exist, so the comparison was
re-measured on one identical set of 4071 held-out crops (2990 HaGRID test crops plus 1081
target-condition crops, role `test` under the current source-based split):

| model | AUC | mean P(hand) on clutter crops |
|---|---|---|
| unfiltered (earlier round) | 0.8597 | 0.4368 |
| arbitered, released | 0.9586 | 0.1967 |
| arbitered, current run seed 0 | 0.8997 | 0.2978 |

The report now quotes the first two, on the same crops with the same loader
(`arbiter_ablation.py`, `results/arbiter_ablation.txt`). The direction is unchanged and the
new figure is if anything conservative, because the unfiltered model predates the
source-based split and may have seen some of these crops. The 41.6 % and 41.9 % arbiter
error rates are carried over from the earlier round unchanged; they measure the arbiter
against the landmark presence head and are not affected by anything changed here.

## 18. Multi-seed campaign stopped early: what is reported, and what is not

The plan was ten seeds for each of the four verifier variants and ten for the detector
fine-tune, with a per-seed end-to-end pipeline evaluation after it. The campaign was stopped
before it finished, so what exists is what is reported:

| variant | seeds completed | held-out HaGRID AUC | target-condition AUC |
|---|---|---|---|
| 11k mined | 10 | 0.9746 ± 0.0046 | not in this pool |
| 130k arbitered | 3 | 0.9507 ± 0.0077 | 0.6439 ± 0.0708 |
| target-condition | 9 | 0.9717 ± 0.0061 | 0.7148 ± 0.0778 |
| 84k dark-augmented | 0 | not run | not run |
| detector fine-tune | 5 | n/a | containment 45.5 % ± 2.0 |

**Not done, and said so in the report:** the per-seed pipeline evaluation
(`seeds_pipeline_eval.py --phase select` then `--phase eval`), which is what would turn the
seeds into mean and SD for per-clip F1, aggregate F1, precision, recall and the false-lock
rate. Section V-D reports held-out AUC variance instead of end-to-end F1 variance and states
that this is a narrower claim than the one planned. The threshold selection in that section
is the per-variant threshold chosen at a fixed seed, not a per-seed selection.

**One checkpoint was discarded.** The target-condition variant was retrained on its full
pool (section 15) but the refit was stopped after writing seed 0, which would have left one
checkpoint trained on 32,027 crops next to nine trained on 11,840. `verifier_cnn_vll.pt_seed0`
and its record were deleted so the nine remaining seeds share one training pool, and the
table says nine seeds. The train_vll curves used in Fig. 1 come from the nine-seed set.

**The confusion-matrix figure is not in the report.** The per-seed records store decisions
on each variant's own held-out pool, and those pools differ (HaGRID test crops for some
variants, HaGRID plus target-condition crops for others), so a single averaged confusion
matrix would mix populations inside one cell. It is generated anyway
(`results/report_figures/fig_confusion_verifier.png`) but it is not used, and a clean
target-condition confusion matrix is listed as not done.

**Selection instability is a result, not a nuisance.** The threshold chosen on the
151-frame validation set ranges from 0.05 to 0.50 across seeds (SD 0.06 to 0.16 within a
variant). That is reported in V-D as a limitation of the validation set, because it means
every threshold in this report is one draw from a wide interval.

**Two stale files deleted.** `results/verifier_seeds_summary.csv` and `.txt` were written by
an earlier one-seed smoke run; the campaign that would have overwritten them was stopped
before its summary step, so they described a single seed per variant and would have been
read as the multi-seed result. They are removed. The authoritative record is
`results/verifier_seeds/<variant>_seed<k>.json`, one file per completed run, which is what
`report_tables.py` and `make_latex_tables.py` read.

## 19. Wording fixed where the arithmetic did not support the sentence

* "my best configuration more than doubles its F1" (target condition) became the two
  measured numbers, 10.4 % to 21.5 %.
* "roughly doubles the baseline's F1" in the Conclusions was wrong: the best configuration
  is far below MediaPipe on that clip. It now reads 10.4 % → 21.5 % against MediaPipe's
  65.1 %.
* The parenthetical in the attribution paragraph that called 37 % of detections "bimodal"
  now quotes both measured shares (36.9 % within 0.35 box widths, 23.8 % beyond two).
