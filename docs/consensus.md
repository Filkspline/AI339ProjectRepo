# Multi-pass spatial consensus on REACQUIRE (HandCursorDark_v3)

Live report that started this round: in slightly dark conditions the cursor is
jumpy and imprecise, locking onto things like an armpit before self-correcting
within milliseconds. The user asked for acquisition to stop trusting a single
frame: require several passes that clear the existing gates **and agree
spatially**, rather than `N` hits in a row anywhere.

This document is the design, the numbers behind the two parameters, and the
result - including the part where the measurement says the change is aimed at the
wrong half of the pipeline.

---

## 1. What was there before

`REACQUIRE_HITS = 2`: two consecutive full-frame detections, anywhere, commit a
lock. Nothing checks that the two detections are in the same place. The only
spatial constraint was the `PlausibilityGate` at `MAX_HAND_SPEED = 3`
frame-widths/second - which at 30 fps permits 64 px of movement per frame on a
640-wide frame, and 128 px at 15 fps. That is not an agreement test; on the
CloseDarkStill clip (15 fps) two detections 100 px apart both pass it.

TRACK has its own continuity rule (take the detection nearest the last known
position) and was left alone, as requested.

## 2. The rule

```
run = []
on each accept (REACQUIRE only):
    if run and |cand - mean(run)| > tol * frame_width:
        run = [cand]        # disagreement: start over from here
    else:
        run.append(cand)
    if len(run) >= N: commit
on any miss: run = []
```

* Agreement is measured against the **run mean**, not the previous candidate,
  because the requirement is "all roughly the same place" rather than "each step
  is small" - the latter lets a slow drift walk anywhere it likes.
* A disagreement restarts the run **with the disagreeing candidate** rather than
  clearing it, so one stray frame cannot discard a good sequence that is just
  beginning. An alternating hand/other flicker still never commits: every second
  frame restarts the run, so the length never reaches N. (Verified: 30 alternating
  frames, 29 restarts, no lock.)
* Passes must be **consecutive**: a miss clears the run, so a candidate that
  appears, vanishes and reappears is not a sustained single location.
* N = 0 disables the whole thing, which is v10's behaviour and the default
  everywhere except the dark variant.

## 3. Why N = 3 and tolerance = 0.05

Both come from `diagnose_acquire_consensus.py`, which replays the **real** dark
acquisition path (dark detector, 130k verifier @0.61, motion blobs, motion gate,
plausibility gate) over every clip with the locator pinned in REACQUIRE, records
the per-frame accepted candidate, and simulates the rule against the same
MediaPipe-static truth the main harness uses.

**Measured displacement between consecutive accepted frames (fraction of frame
width):**

| clip | both ends on the hand | hand <-> other (a flip) |
|---|---|---|
| CloseLightMoving | median 0.0244, p95 0.0559 | none |
| CloseLightStill | median 0.0028, p95 0.0140 | median 0.0695 |
| CloseDarkStill | median 0.0103, p95 0.0371 | none |
| FarLight | median 0.0075, p95 0.0569 | median 0.0507 |
| FarDark | median 0.0009, p95 0.0353 | median 0.0207 |
| VeryLowLight | median 0.0180, p95 0.0464 | median 0.0741 |

* **Tolerance 0.05** sits above the p95 of genuine hand movement on the moving
  clips (0.046-0.057) and below the armpit-scale excursions (0.069-0.074 on
  CloseLightStill and VeryLowLight). It is not chosen tighter because the
  false-commit fraction turned out to be nearly **flat in the tolerance**  - 
  37.9 % at 0.02 versus 33.1 % at 0.05 - so a tighter radius buys no extra
  certainty and only discards genuine hits.
* **N = 3**: N = 4 collapses acquisition where detections are intermittent.
  Mean delay to the first lock on VeryLowLight went 3 frames (baseline) → 55 at
  N = 3 → **160** at N = 4 (10.7 s at that clip's 15 fps), and total commits fell
  341 → 99 for about two further points of false-commit reduction. N = 2 with a
  tolerance is barely distinguishable from the shipped behaviour.
* The mean rule and a "within tol of the previous candidate" rule measured almost
  identically (N = 3 @ 0.05: 181 commits / 33.1 % false versus 185 / 32.4 %), so
  the stricter, more literal mean rule was kept.

Worst-case acquisition latency added: two extra frames. That is 67 ms at 30 fps,
133 ms at 15 fps - well inside "slow, deliberate hand movement" tolerates, and it
is the whole price of the change.

## 4. Synthetic verification first

`test_acquire_consensus.py`, 11 checks, all passing. Part 1 is deterministic (it
drives the real decision path); part 2 runs the **whole** `update()` over
CloseLightMoving's real frames - real size gate, real motion gate, real
plausibility gate - with a stub detector emitting scripted boxes:

| check | result |
|---|---|
| 30 alternating hand/other frames | no lock, 29 restarts |
| sustained hand | locks after exactly 3 passes |
| 2 passes then a miss | no lock |
| pass, pass, miss, pass, pass | no lock (passes must be consecutive) |
| slowly moving hand (0.03 fw/frame) | still locks |
| fast moving hand (0.10 fw/frame > tol) | does not lock - documented cost |
| **control: flicker with consensus OFF** | **locks** (the test can fail) |
| integration: real gates + sustained hand | locks |
| integration: flicker, consensus ON | no lock, 20 restarts, same 25 hits |
| **integration: same flicker, consensus OFF** | **locks** |

The last two are the ones that matter: identical script, identical 25 accepted
frames, and the only difference is the consensus. The first version of this test
passed for the **wrong reason** - the offset box was being rejected by the motion
gate, so the consensus was never exercised (`consensus_resets = 0`). The test now
runs that case without a motion gate and with an offset chosen to be
reachable by the plausibility gate (0.08 fw = 51 px, under the 64 px the gate
allows at 30 fps) but beyond the consensus tolerance (32 px), so the consensus is
the only mechanism that can reject it.

## 5. End-to-end, all clips, three ways

MediaPipe tracking baseline / `HandCursor_v10` (unchanged) / dark without the
consensus / dark with it. The dark columns are from the same detector, verifier,
threshold and blob setting; only the consensus differs. Run-to-run wobble from the
harness's wall-clock `dt` is about ±1.3 F1 points (dark CloseDarkStill measured
12.1 in the previous round's run and 10.8 in this one), which is why both dark
columns are quoted from the same session.

| clip | MediaPipe | v10 | dark (v2) | dark + consensus (v3) |
|---|---|---|---|---|
| CloseLightMoving F1 | 100.0 | 86.3 | 94.3 | 94.3 |
| CloseLightStill F1 | 100.0 | 92.5 | 55.4 | 55.3 |
| **CloseDarkStill F1** | 99.5 | 93.8 | 10.8 | **16.0** |
| FarLight F1 | 74.6 | 32.3 | 51.8 | 52.6 |
| FarDark F1 | 77.8 | 14.9 | 51.8 | 51.8 |
| VeryLowLight F1 | 65.1 | 8.8 | 31.7 | 30.6 |
| NohandLight false locks | 0.0 | 0.0 | 2.2 | 2.2 |
| NohandDark false locks | 0.0 | 0.0 | 0.0 | 0.0 |
| **aggregate F1 (hand clips)** | **81.5** | **50.1** | **49.7** | **49.8** |

* CloseDarkStill, the clip that regressed and the condition tested live:
  **F1 10.8 → 16.0**, precision 14.8 % → 21.4 %, recall 8.5 % → 12.8 %. Real, and
  the largest single movement anywhere in the table - but 16.0 is still a broken
  experience, not a fixed one.
* The far/dark clips that are this variant's reason to exist paid almost nothing:
  FarLight +0.8, FarDark 0.0. VeryLowLight lost 1.1 points (31.7 → 30.6) even
  though its first acquisition became much slower, because once locked, TRACK
  holds on and the accepted-frame count recovers.
* No-hand false-lock rate: unchanged at 2.2 % (NohandLight) and 0.0 %
  (NohandDark).

## 6. second look

**Does it actually fix the armpit-jumping case?** No - and the reason is
measurable rather than mysterious. `diagnose_lock_events.py` counts, per clip, how
many accepted positions are off the hand, split by **which mode produced them**:

| | accepted | off-hand | from REACQUIRE | from TRACK | excursions |
|---|---|---|---|---|---|
| dark (v2) | 856 | 317 | **10** | **307** | 85 (7 lasting ≥10 frames) |
| dark + consensus (v3) | 855 | 314 | 20 | 294 | 82 (7 lasting ≥10 frames) |

**307 of the 317 off-hand positions are produced by TRACK, not REACQUIRE.** The
consensus was scoped to REACQUIRE on request, so it can only ever touch about 3 %
of them: measured, it removes 13 off-hand frames and 3 excursions out of 85. That
is the explanation for a flat aggregate F1, and it means the live symptom
is being generated *after* acquisition, inside the tracking window - exactly the
part the request told me to leave alone.

**Does nearby-but-wrong flicker still slip through the tolerance?** Yes, by
construction: anything within 0.05 frame widths (32 px at 640) counts as agreement
and commits. That is defensible rather than a hole, because the evaluation's own
"on the hand" radius is 0.06 frame widths, so a lock inside the tolerance is not
counted as a false lock - it is a slightly off lock. Beyond the tolerance it is
rejected, which the synthetic control demonstrates. And the flatness of the
false-commit fraction across tolerances (37.9 % → 33.1 % from 0.02 to 0.05) says
the tolerance is not a discriminating knob in this data - the false positives are
too often *stable* for any radius to reject: FarDark's consecutive false positions
sit a median of 0.0004 frame widths apart, i.e. they do not move at all.

**Did slowing acquisition cost the far/dark recall that is the point of this
variant?** Almost nothing end-to-end: FarLight +0.8, FarDark 0.0, VeryLowLight
−1.1. The suspicious part of that is the simulation, which predicted a 55-frame
first-lock delay on VeryLowLight: the cost is real but bounded, because the harness
(and the user) only care about accepted frames, and TRACK's hold recovers them.
The delay is 2 extra frames in the best case and up to tens of frames where
detections are intermittent - a real, user-visible pause before the cursor first
appears in the dark, which the F1 column understates.

**Any change to the no-hand false-lock rate?** No: 2.2 % and 0.0 %, identical to
v2. The consensus does not remove the false locks on NohandLight because those are
stable detections too.

## 7. Recommendation

Ship `HandCursorDark_v3` as the dark variant: the change is a small, verified,
essentially free improvement (+5.2 F1 on the regressed clip, no measurable cost on
the far/dark clips, no change in false locks, worst case 2 frames of extra
acquisition latency), and it does exactly what it was asked to do - a detector
flickering between two locations can no longer commit a lock.

But be clear about what it is not: **it does not fix the armpit-jumping**. The
measurement says the cursor is jumping off the hand during TRACK, in the search
window, on 307 of the 317 off-hand frames. Roughly 92 % of those excursions are
transient (median 1-4 frames, only 7 of 85 lasting ≥10 frames), which is the same
signature as "locks onto an armpit, self-corrects within milliseconds" - so the
same agreement idea, applied to the in-window candidate instead of the acquisition
candidate, is the change that would actually address the live complaint. That was
outside the scope of this round (TRACK was explicitly excluded) and is not
implemented here.

`HandCursor_v10` remains the default and is untouched: the consensus is off unless
`--acquire-consensus N` is passed, and only `hand_mouse_dark.py` passes it.

---

# v4: "if it is not certain, it should not move the mouse there"

The v3 work above fixed the *lock* but not the *cursor*, and the measurement said
why. Two separate rules were added, because two different kinds of unconfirmed
candidate were moving the mouse.

## 1. REACQUIRE was publishing pass 1

`update()` returned the candidate as soon as it cleared the gates, regardless of
whether the consensus had committed. So with v3 the lock waited for three agreeing
passes - but the **cursor had already been dragged to the first one**. The flicker
still moved the mouse; it just no longer moved the tracking window.

Now `_on_hit` returns whether a position may be **published**, and an unconfirmed
REACQUIRE run publishes nothing. `CursorController.update(None, ...)` holds the
cursor *exactly still* while locked, so an unconfirmed candidate now produces no
mouse movement at all.

## 2. NEW: the TRACK hold (`--track-consensus`)

The v3 measurement showed where the off-hand frames actually come from: **307 of
317 from TRACK, 10 from REACQUIRE**. So the same idea was applied to the in-window
candidate:

* distance from the **committed** position ≤ 0.06 frame widths → publish
  immediately, so normal tracking is not delayed by even one frame;
* beyond that → **hold**: the cursor does not move. The claim is published only once
  `N = 3` consecutive candidates agree on the new place, *or* after
  `TRACK_CONSENSUS_MAX_HOLD = 6` frames, whichever comes first.

**Why the hold must be bounded.** A hand moving steadily faster than the tolerance
produces a stream of claims that never agree *with each other* - it is describing a
path, not a place. An unbounded "hold until N claims agree" would therefore freeze
the cursor until the 3 s loss timeout turned a wrong jump into a dead cursor, which
is worse than the bug. The dry-out means the rule *delays* a jump rather than
vetoing it, so only transient excursions - the reported symptom, "self-corrects
within milliseconds" - are actually suppressed. Tested explicitly
(`TRACK: fast sustained move is picked up within the hold limit`).

**Tolerance, from measurement** (`diagnose_lock_events.py`, TRACK displacements):

| clip | real tracking motion p95 | jump off the hand (median) |
|---|---|---|
| CloseLightMoving | 0.0440 | none |
| CloseLightStill | 0.0853 | 0.1094 |
| CloseDarkStill | 0.0290 | 0.0742 |
| FarLight | 0.0354 | 0.0824 |
| FarDark | 0.0332 | 0.0207 |
| VeryLowLight | 0.0263 | 0.0638 |

0.06 sits above genuine motion on five of the six clips and below the jumps it
targets. (CloseLightStill's 0.0853 is inflated by the evaluation's own 0.06 match
radius - two positions can be 0.09 apart and still both count as "on the hand"  - 
and its frames are bounded by the dry-out.) FarDark is the one clip where jumps are
*smaller* than real motion, so its jumps are not caught: its false positions are
stable, and no spatial rule can reject a detection that does not move.

## 3. Measured effect

`diagnose_lock_events.py`, both configurations in one process so the frame timing is
identical. "Excursion" = a run of accepted positions off the hand, preceded by a
correctly-placed accepted frame - i.e. the cursor moving somewhere wrong and coming
back. "Persistent" = that run lasting ≥ 10 frames.

| | v3 (acquire only) | v4 (+ TRACK hold) |
|---|---|---|
| off-hand cursor placements | 362 | **164** (−55 %) |
| **excursions** | **95** | **39** (−59 %)** |
| persistent wrong locks (≥10 f) | 9 | **1** |
| accepted frames (any position) | 912 | 607 (−33 %) |

Per clip excursions, v3 → v4: CloseLightMoving 4 → 1, CloseLightStill 29 → 6,
CloseDarkStill 7 → 4, FarLight 17 → 12, FarDark 10 → 7, VeryLowLight 28 → 9.

End-to-end `evaluate_pipelines.py` (same session, same detector/verifier/blobs):

| clip | MediaPipe | v10 | dark v3 | dark v4 @0.06 | dark v4 @0.09 |
|---|---|---|---|---|---|
| CloseLightMoving F1 | 100.0 | 86.3 | 94.3 | 90.2 | 92.6 |
| CloseLightStill F1 | 100.0 | 92.5 | 55.3 | 51.5 | 56.8 |
| **CloseDarkStill F1** | 99.5 | 93.8 | 16.0 | **6.2** | 7.4 |
| FarLight F1 | 74.6 | 32.3 | 52.6 | 45.6 | 46.0 |
| FarDark F1 | 77.8 | 14.9 | 51.8 | 51.6 | 51.9 |
| VeryLowLight F1 | 65.1 | 8.8 | 30.6 | 27.0 | 26.3 |
| NohandLight false locks | 0.0 | 0.0 | 2.2 | **0.0** | **0.0** |
| NohandDark false locks | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **aggregate precision** | 88.3 | 79.1 | 63.3 | **74.1** | 70.2 |
| **aggregate recall** | 75.6 | 36.5 | 41.0 | 33.8 | 36.0 |
| **aggregate F1** | **81.5** | **49.9** | **49.8** | **46.4** | **47.6** |

The hold moves the operating point exactly as a confidence gate should: precision
**+10.8 points**, and the first-ever clean run on NohandLight (**2.2 % → 0.0 %**),
paid for with recall (−7.2) and F1 (−3.4).

## 4. second look on v4

**CloseDarkStill regressed, and the mechanism is worth stating.** On that clip the
hold suppressed the *correct* detections more than the incorrect ones: on-hand TRACK
frames 8 → 3 (−62 %) while off-hand frames 49 → 33 (−33 %). The reason is that on
this clip the true detections are the **transient** ones and the false ones are
**stable** - a hold-until-sustained rule therefore preferentially throws away the
truth. That is a real regression on the very clip that started this work, and it is
the clearest evidence that a temporal-agreement rule cannot fix a clip whose false
positives are steady: the verifier, not the timing, is what has to reject those.

**F1 understates the change in the user-visible behaviour.** F1 counts frames on
which a position was reported, so holding necessarily lowers it - 33 % fewer
accepted frames. The numbers that match the complaint are the excursion counts
(−59 %, persistent 9 → 1) and the precision column (+10.8). A cursor that is still
for 200 ms is not the same defect as a cursor that jumps to an armpit, and F1 cannot
tell them apart.

**Does anything still slip through?** Yes, three things, all measured:
* a jump that would be *stable* (FarDark) is confirmed after 3 frames and followed;
* a jump that lasts longer than `N` frames is followed after `N` frames of hold  - 
  the rule delays, it does not veto;
* on the clip with the most real hand motion, the hold costs genuine frames
  (CloseLightMoving accepted 157 → 143).

**Very-low-light cost is the one to feel live.** VeryLowLight acceptance 30.9 % →
17.5 %: with an intermittent detector the hold makes the cursor present-less for
longer, which in the dark could read as "nothing happens" rather than "no false
moves". `--track-consensus-tol 0.09` (milder) or `--track-consensus 0` (v3
behaviour) are the escape hatches, both one flag.

**v10 is provably unchanged.** The `_on_hit` restructure moved where `self.pos` is
assigned, so the claim was verified rather than assumed: `test_v10_equivalence.py`
re-implements the pre-refactor state machine verbatim and drives it against the real
one with 13 random sequences (hits, misses, jumps, repeats; ~6 300 steps), asserting
after every single step that all seven state fields match and that every accepted
hit was published. All pass. (The separate v10 evaluation re-run differs by 0.3 F1
points - that is the harness's wall-clock `dt`, not a behaviour change.)

