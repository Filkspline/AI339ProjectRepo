# Gate replacement design (wired into the cursor app)

How the learned hand-vs-clutter classifier replaces the box-size FP gate.
Originally written as a design so the swap would be a **parameter change** rather
than a rewrite; it is now implemented in `hand_mouse_cursor.py` (TRACK mode) and
the measured outcome is in the last section.

## Status (implemented)

| piece | where | state |
|---|---|---|
| `SmallCNN` + `verify(crop_bgr) -> P(hand)` | `crop_verifier.py` | done |
| training set (HaGRID + our own clips) | `build_verifier_dataset.py` | 11203 crops |
| CNN training / comparison | `train_cnn_verifier.py` | `verifier_cnn.pt`, ~50k params |
| TRACK-mode gate in the locator | `hand_mouse_cursor.py`, `detect_window(verifier=...)` | done, `--no-verifier` reverts |
| bundled weights | `build_release.py` (`ROOT_DATA`) | done |
| `HandTracker.process` in `hand_pipeline.py` | - | **not** done (see below) |

Note the integration point ended up in the locator's windowed front-end rather
than in `HandTracker.process`: the clutter problem is specific to TRACK mode,
where the in-window size gate is loose (0.04), and the cursor app
does not run the landmark model at all. `HandTracker` (used by
`hand_mouse_pytorch.py`) still has no verifier hook; the code below is the recipe
if it is wanted there.

## Why the box-size gate has to go

The gate is `box_width >= 0.15 * frame_width` (`HandTracker.process`). It works
by assuming "far/clutter = small box". Measured consequences:

| clip | gate threshold | measured boxW | outcome |
|---|---|---|---|
| Close* | 96 px (640) | ~324 px | kept |
| FarLight | 288 px (1920) | 136 px | **rejected** |
| FarDark | 192 px (1280) | 116 px | **rejected** |
| NohandLight (clutter) | 96 px | 60-100 px | rejected *by luck* |

It rejects far hands (the thing we now need) and only rejects clutter
*coincidentally*, because clutter happened to be small. A learned classifier
attacks the actual question - "is this crop a hand?" - at any size.

## Interface

Add two constructor parameters; nothing else changes shape.

```python
class HandTracker:
    def __init__(self, ml_dir=None, min_score=0.5, min_box_width=0.15,
                 verifier=None,                 # callable(crop_bgr) -> P(hand)
                 verifier_threshold=0.5):
        self.min_score = min_score
        self.min_box_width = min_box_width
        self.verifier = verifier
        self.verifier_threshold = verifier_threshold
```

* `verifier` is a **plain callable taking the 256×256 BGR crop** and returning a
  probability in `[0, 1]`. not a class - so the cheap baseline, a
  small CNN, or a segmentation-derived score can all be dropped in unchanged.
* `verifier is None` → **exactly today's behaviour** (box-size gate). The swap is
  opt-in and reversible, so the live pipeline keeps working throughout.

## Integration point

Inside `HandTracker.process`, the crop is already computed *before* the landmark
model runs. Verification slots in there - after the crop, before the landmark
head - so rejected crops also skip landmark inference (net compute saving at
range, where most clutter lives):

```python
for det in detections[0]:
    score = float(det[18])
    if score < self.min_score:
        continue
    box = box_to_pixels(det[:4].tolist(), scale, left, top)
    kps = [...]
    cx, cy, side, rotation = hand_roi(box, kps[0], kps[2])
    rect = rotated_rect_to_points(cx, cy, side, side, rotation)
    crop, M = warp_rect(frame_bgr, rect, 256)

    # ---- detection acceptance (the only changed block) ----
    if self.verifier is not None:
        p_hand = float(self.verifier(crop))       # learned hand-vs-clutter
        if p_hand < self.verifier_threshold:
            continue
    else:
        xmin, ymin, xmax, ymax = box             # legacy box-size gate
        if (xmax - xmin) < self.min_box_width * W:
            continue
    # -------------------------------------------------------

    # landmark model only runs on accepted crops
    ...
    hands.append(dict(box=box, score=score, p_hand=p_hand, ...))
```

Notes:
* `p_hand` is surfaced in the returned dict so threshold tuning and debugging do
  not require re-instrumenting the pipeline.
* The verifier gets the **same canonical rotation-normalised crop** the
  classifier was trained on (Part 1's crop), so no train/serve skew.
* Multiple detections per frame are scored independently - an improvement over
  the gate, which could only reason about size.

## Wiring a verifier in (call site, later)

```python
from baseline_classifier import BaselineClassifier, crop_features

def make_verifier(model_path):
    clf = BaselineClassifier.load(model_path)
    def verify(crop_bgr):
        feat = crop_features(cv2.resize(crop_bgr, (128, 128),
                                        interpolation=cv2.INTER_AREA))
        return float(clf.predict_proba(feat[None, :])[0])
    return verify

tracker = HandTracker(ml_dir=MODEL_DIR, verifier=make_verifier("baseline.pt"))
```

## Coexistence, not just replacement

The verifier answers **"is this a hand?"**; it does not answer "is it open?".
The second head is the same crop, same features/CNN - so a single model should
eventually emit both:

| output | consumer |
|---|---|
| `P(hand)` | replaces the box-size gate (above) |
| `P(open)` | the open/closed requirement |

Decisions deferred until data exists:
* whether clutter rejection and open/fist share one trunk (likely yes - clutter
  is just a third class, or a separate sigmoid head).
* whether to keep `presence` (landmark hand-flag) as a second opinion. It was a
  strong hand/clutter signal at close range (0.85-1.00 vs ~0.03 for clutter) but
  collapses to ~0.00 for real far/dark hands, so it cannot replace the gate on
  its own - at most an ensemble member.
* the operating threshold: pick it on **held-out scenes** optimising
  clutter-rejection at a fixed recall for real hands, since a false positive and
  a missed far hand are not symmetric costs.

## Rollback

If the verifier underperforms, passing `verifier=None` (or running the app with
`--no-verifier`) restores the current behaviour with no other edits - the gate
code is retained, not deleted.

## Measured outcome

Evaluated by replaying the recorded clips through the locator twice
(`eval_verifier_vs_gt.py`), labelling every accepted frame against MediaPipe's
own hands solution (`gt_clips.py`): a frame is a **hit** if the position is
within a hand-scaled radius of MediaPipe's palm centre (borrowing the nearest
detection within ±5 frames, because MediaPipe itself only sees the far hand in
~60% of frames), otherwise it is a **clutter lock**.

| TRACK gate | true hand frames | clutter locks |
|---|---|---|
| 0.04 box-size gate (v2) | 499 | 156 |
| logistic 8-feature baseline (thr 0.30) | 379 | 190 |
| verifier thr 0.20 | 490 | 143 |
| **verifier thr 0.30** | **486** | **32** |
| verifier thr 0.40 | 471 | 12 |
| verifier thr 0.50 | 461 | 9 |

The small CNN is not a marginal upgrade over the cheap baseline - the logistic
classifier is worse than leaving the gate alone (it loses 120 hand frames *and*
adds 34 clutter locks, failing completely on the dark and far clips). Its
HaGRID held-out AUC was 0.7884 against the CNN's 0.9650, and its 8 hand-designed
crop features (foreground fraction, radial moments, edge density, intensity
moments) evidently do not survive the dark/upscaled regime.

Per clip, v2 vs v3 (thr 0.30), hit / FP:
CloseLightStill 209/0 → 209/0, CloseDarkStill 90/0 → 87/0,
FarLight 32/56 → 25/26, FarDark 1/96 → 6/3.

Two findings worth keeping:

1. **The v2 "range gain" was mostly false positives.** FarDark accepted 42% of
   frames under the size gate, but only 1 of those 97 accepted frames was within
   the hand's location - 96 were locks on room clutter. FarLight was 32 hit / 56
   clutter. The relaxed gate was not giving us range; it was padding the
   accepted-frame rate with clutter, which is exactly what live testing reported.
2. **The verifier keeps the recall and removes the clutter.** At 0.30 it keeps
   486 of 499 true hand frames (97.4%) while cutting clutter locks from 156 to
   32 (−79%).

The weak point is that our own-room training data is lopsided: the 77 own-room
hand crops are all close-up (box width 0.06-0.56 of frame, median 0.47) while the
23 own-room clutter crops are all small (0.044-0.10). The classifier never sees
box size (the crop is warp-normalised), so this biases it through image
sharpness. Simulating distance on the stored crops (downscale to 1/10 then back
up) only moves P(hand) from 0.658 to 0.490, so the shortcut is real but not
dominant. More own-room far positives would be the fix.

Not covered by the verifier: REACQUIRE mode, which still uses the 0.15 full-frame
gate. All remaining no-hand false positives are there (2 frames on NohandDark,
which v1 also had), since NohandDark never enters TRACK at all.
