# Reproducing the report

This is the analysis half of the project: every number in `AIML339_Final_Report.tex` comes
from a command in this file. The application half (the cursor itself, the packaged builds,
the design documents) is described in `README.md` and `docs/`.

Nothing here is hand-edited. Every table and figure is written by a script from measured
output, and every script writes its result files under `results/`.

## 1. Environment

Two virtual environments, because the two halves of the pipeline need libraries that do not
coexist in this project's pins.

| environment | packages | used for |
|---|---|---|
| `.venv-blazepalm` | torch 2.13.0+cpu, opencv 5.0.0, numpy 2.2.6, pyarrow 25.0.1 | the detector, the verifiers, all pipeline evaluation, the multi-seed runs |
| `.venv` | mediapipe 0.10.14, opencv 5.0.0, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9 | the MediaPipe baseline and ground truth, the statistical tests, the figures |

Versions as installed and used for every number in the report (printed from the two
environments). Neither environment contains the other's key package, which is why the
project uses two.

`requirements.txt` pins the application half. The analysis half has no separate pin file;
install into a fresh environment with:

```powershell
python -m venv .venv-blazepalm
.\.venv-blazepalm\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-blazepalm\Scripts\python.exe -m pip install opencv-python numpy pyarrow

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install mediapipe opencv-python numpy scipy matplotlib
```

Hardware used: one AMD Ryzen 5 7500F desktop, CPU only, no GPU, Windows. Nothing here needs
a GPU, and no result depends on the machine being fast: evaluation timing is derived from
the frame index and each clip's own frame rate, so a slower machine gives the same numbers
(see `check_determinism.py`).

## 2. Data preparation

The eight clips (`CloseLightMoving.mp4`, `CloseLightStill.mp4`, `CloseDarkStill.mp4`,
`FarLight.mp4`, `FarDark.mp4`, `NohandLight.mp4`, `NohandDark.mp4`, `VeryLowLight.mp4`) are
the project author's own footage and are in the repository root.

HaGRID and HaGRIDv2 are third-party data and are **not** redistributed here. Fetch and
rebuild the crop sets with:

```powershell
.\.venv-blazepalm\Scripts\python.exe fetch_hagrid.py            # download the 512px release
.\.venv-blazepalm\Scripts\python.exe build_verifier_dataset.py  # 11,203 crops  -> verifier_dataset/
.\.venv-blazepalm\Scripts\python.exe build_vll_dataset.py       # 32,027 crops  -> vll_dataset/
.\.venv-blazepalm\Scripts\python.exe build_big_dataset.py       # 130,532 crops -> big_dataset/
.\.venv-blazepalm\Scripts\python.exe build_dark_dataset.py      # 84,670 crops  -> dark_dataset/
```

The ground truth is generated from MediaPipe in static mode and stored per clip:

```powershell
.\.venv\Scripts\python.exe gt_clips_static.py        # -> gt_clips.npz
.\.venv\Scripts\python.exe gt_verylowlight.py        # -> gt_verylowlight.npz (several confidences
                                                     #    and both modes for the target clip)
```

The VeryLowLight three-way split is fixed by seed and written once:

```powershell
.\.venv-blazepalm\Scripts\python.exe vll_split.py    # -> results/vll_split.json
```

Split seed 0, 50-frame blocks: train blocks 0, 2, 3, 4, 5 (250 frames), validation blocks
7, 10, 11, 13 (151 frames), test blocks 1, 6, 8, 9, 12 (250 frames). Every selection in the
report uses validation only; the test blocks are evaluated once.

## 3. Training

All seeds are fixed and shared: **split seed 0, training seeds 0-9, bootstrap seed 12345**.
No code path reads the clock.

```powershell
# verifier, four variants x ten seeds (this is the long one: ~1 h per large variant)
.\.venv-blazepalm\Scripts\python.exe train_verifier_seeds.py --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 15

# detector fine-tune (six prediction heads only), ten seeds
.\.venv-blazepalm\Scripts\python.exe finetune_detector_seeds.py --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 15
```

The checkpoint kept per seed is the epoch with the lowest validation loss. The operating
threshold is *not* chosen here: it is chosen once per variant on the validation blocks in
step 4, so a seed changes the weights only and the seed-to-seed spread is not inflated by
threshold selection. Per-run records (parameters, training seconds, selected epoch,
validation score, held-out AUC, the threshold-set F1, and for the detector the collapse
guard result) are written to `results/verifier_seeds/<variant>_seed<k>.json` and
`results/detector_seeds/detector_seed<k>.json`, with the epoch curves in
`results/verifier_seeds_curves_<variant>.csv` and `results/detector_seeds_curves.csv`.

The collapse guard is the one that refuses to save a fine-tuned detector scoring above 0.5
on more than 100 anchors of a light frame. It is reported per seed, not silently applied.

## 4. Evaluation

```powershell
# threshold per variant, on VeryLowLight validation blocks only -> results/thresholds.json
.\.venv-blazepalm\Scripts\python.exe seeds_pipeline_eval.py --phase select

# one session, all seven systems, identical truth -> results/summary_all_systems.csv
.\.venv-blazepalm\Scripts\python.exe evaluate_all_systems.py

# per seed, every clip -> results/seed_eval_<variant>.csv, results/seed_per_frame/
.\.venv-blazepalm\Scripts\python.exe seeds_pipeline_eval.py --phase eval
```

`evaluate_all_systems.py` is the table generator behind Table I: MediaPipe in tracking mode,
two contrast baselines in front of unmodified MediaPipe, and the project's own
configurations, all in one process with one truth, so no system is flattered by a different
run. It also writes `results/per_frame_<system>.csv`, which is what the paired tests read.

Attribution of the loss to pipeline stages, on the target clip:

```powershell
.\.venv-blazepalm\Scripts\python.exe diagnose_verylowlight.py --verifier verifier_cnn.pt --threshold 0.30 --gt-key static_conf0.3 --motion-blobs
.\.venv-blazepalm\Scripts\python.exe diagnose_verylowlight.py --verifier verifier_cnn_big.pt --threshold 0.61 --gt-key static_conf0.3 --motion-blobs
```

Consensus-rule effect, both configurations in one process:

```powershell
.\.venv-blazepalm\Scripts\python.exe diagnose_lock_events.py
```

Determinism check (the same configuration three times, ignoring the milliseconds column):

```powershell
.\.venv-blazepalm\Scripts\python.exe evaluate_pipelines.py --out-csv build\det_run1.csv --out-txt build\det_run1.txt
.\.venv-blazepalm\Scripts\python.exe check_determinism.py
```

## 5. Statistics and figures

```powershell
.\.venv\Scripts\python.exe stats_tests.py       # -> results/stats_paired_frames.csv, stats_meta.json
.\.venv\Scripts\python.exe plots_report.py      # -> results/report_figures/*.png
```

`stats_tests.py` reports two scopes: `hand` (the six hand clips, the scope the report
quotes) and `all`, and records which is which in the CSV. Tests: exact McNemar on paired
per-frame decisions, 50-frame block bootstrap for F1 differences (2000 resamples, seed
12345), Wilcoxon signed-rank or paired t-test across seeds by Shapiro-Wilk, Holm-Bonferroni
within each family, alpha 0.05.

## 6. Which file produces which table or figure

| Report item | Produced by | Result file |
|---|---|---|
| Table I, all systems in one session | `evaluate_all_systems.py` | `results/summary_all_systems.csv`, printed by `report_tables.py` |
| Table II, per-clip F1 | `evaluate_all_systems.py` | same file, per clip, printed by `clip_table.py` |
| Table III, target clip by split role | `evaluate_all_systems.py` | `results/summary_vll_roles.csv` |
| Table IV, verifier variants over ten seeds | `train_verifier_seeds.py` + `seeds_pipeline_eval.py --phase eval` | `results/verifier_seeds/*.json`, `results/seed_eval_<variant>_aggregate.csv` |
| Table V, detector fine-tune over ten seeds | `finetune_detector_seeds.py` + `seeds_pipeline_eval.py --phase eval` | `results/detector_seeds/*.json`, `results/seed_eval_detector_seed_aggregate.csv` |
| Table VI, paired tests | `stats_tests.py` | `results/stats_paired_frames.csv` |
| Fig. 1, verifier convergence | `train_verifier_seeds.py` | `results/verifier_seeds_curves_<variant>.csv` -> `results/report_figures/fig_convergence_verifier.png` |
| Fig. 2, detector convergence | `finetune_detector_seeds.py` | `results/detector_seeds_curves.csv` -> `results/report_figures/fig_convergence_detector.png` |
| Fig. 3, verifier confusion matrix | `train_verifier_seeds.py` | `results/verifier_seeds/*.json` -> `results/report_figures/fig_confusion_verifier.png` |
| Fig. 4, F1 across seeds | `seeds_pipeline_eval.py --phase eval` | `results/seed_eval_*_aggregate.csv` -> `results/report_figures/fig_seed_f1.png` |
| Detector attribution (Sec. V-B) | `diagnose_verylowlight.py` | `build/funnel_static_*.log`, `results/*_per_frame.csv` |
| Consensus-rule effect (Sec. VI) | `diagnose_lock_events.py` | `build/lock_events_new.log` |
| Timing determinism (Sec. IV-B) | `evaluate_pipelines.py` x3 + `check_determinism.py` | `build/det_run*.csv` |
| Ground-truth self-consistency (Sec. V-E) | `gt_selfconsistency.py` | `results/human_labels/gt_selfconsistency.txt` |
| Arbiter ablation (Sec. III-B) | `arbiter_ablation.py` | `results/arbiter_ablation.txt` |

Two conveniences, so no table is transcribed by hand:

```powershell
.\.venv-blazepalm\Scripts\python.exe report_tables.py      # every table as text -> results/report_tables.txt
.\.venv-blazepalm\Scripts\python.exe make_latex_tables.py  # LaTeX for the multi-seed and
                                                           # statistics tables -> results/latex_fragments.tex
```

## 7. Hand labelling, if you want to close the circularity

Every number in the report is scored against MediaPipe, which is the same model family as
the baseline. 150 target-clip test frames were sampled for human labelling and the human
labels were **not** collected, so the report says so and reports only the yardstick's own
self-consistency instead. To collect them:

```powershell
.\.venv\Scripts\python.exe label_frames.py            # interactive: drag a box, n = no hand, u = unsure
.\.venv\Scripts\python.exe label_frames.py --export   # or label offline from the PNGs + template.csv
```

That writes `results/human_labels/human_labels.csv`, which a human-scored comparison can
then be built from. The frame list is `results/human_labels/frames_to_label.txt`.
