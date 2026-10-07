# Motion-blob candidates (v8, OFF by default)

Built because Step 1 of the VeryLowLight diagnosis showed the detector's *box* is
the dominant failure (64.3% of hand frames: the detector fires confidently at a
normal rate but its box is usually not on the hand), while the hand itself is
clearly visible in the frame-difference signal - it is moving.  So: use motion to
PROPOSE candidate regions independently of the detector, and let the CNN verifier
CONFIRM them, exactly as it already does for detector candidates.  No BlazePalm
weights or architecture were touched.

**Verdict: the proposal half works, the confirmation half does not.  Shipped
OFF by default (`--motion-blobs` to enable).**  Numbers below.

## What was built

* `MotionGate.blobs()` - connected components on the existing difference mask
  (`_diff`, reused from the motion gate: same 320-wide INTER_AREA ring, same
  12/255 pixel threshold, same noise characterisation, no re-derivation), filtered
  by `MOTION_BLOB_MIN_SIDE_FRAC = 0.03` of frame width - tied to the established
  range floor (~0.033 of frame width is the smallest hand the window path can
  resolve) and it kills sensor-noise speckles.  Up to `MOTION_BLOB_MAX = 3` blobs
  per frame, largest first, returned in frame pixels.
* `blob_candidates()` - for each blob, a square ROI of side
  `blob_side x MOTION_BLOB_ROI_SCALE`, scored by the verifier at
  `MOTION_BLOB_ROTATIONS = 8` in-plane orientations (a blob has no keypoints, so
  the wrist->MCP rotation the verifier was trained with is unknown; the max wins).
  A blob below `verifier_threshold` is discarded - motion alone never acquires.
* `HandLocator(use_motion_blobs=True)` - REACQUIRE now verifier-checks BOTH the
  detector's candidates and the blobs, and the best above threshold is used
  (`REACQUIRE_HITS` unchanged); TRACK admits in-window blobs and lets one override
  the detector's candidate only if it wins by `MOTION_BLOB_PREFER_MARGIN = 0.10`,
  which protects the existing continuity behaviour.  When the flag is off, the
  code path is byte-identical to v6/v7 (tests still 50/50).

## Calibration (`calibrate_motion_blobs.py`)

**Proposal quality.** On `VeryLowLight.mp4`, a motion blob sits on the hand in
**42.2%** of MediaPipe-hand frames - better than the detector's 35.7%
box-containment rate.  So the proposal idea is sound.

**ROI scale.** Measured `2.6 x palm width / blob side` over 317 blob-on-hand
pairs: median **1.20** (p25 0.75, p75 1.72).  `MOTION_BLOB_ROI_SCALE = 1.2` is set
from that geometry, *not* from which value scored best on the
verifier - because no value scored well.

**Rotation.** P(hand) varies only 1.3-1.5x across all in-plane orientations, so the
unknown blob orientation costs little and the 8-way search is more than enough.
Not the problem.

**Confirmation - this is where it fails.** Blobs labelled with BOTH MediaPipe
modes (using static mode alone would have counted real hand blobs as clutter in
the 25% of frames static mode misses, inflating the false-accept rate):

| ROI scale | on-hand blobs accepted | off-hand blobs accepted | separation |
|---|---|---|---|
| 0.8 | 50.8% | 82.2% | -31.5 |
| 1.0 | 59.0% | 86.2% | -27.3 |
| 1.1 | 67.8% | 92.2% | -24.4 |
| 1.3 | 67.8% | 88.8% | -20.9 |
| 1.6 | 69.4% | 92.5% | -23.1 |
| 2.0 | 76.7% | 94.5% | -17.8 |

The verifier accepts **moving clutter more often than the moving hand** at every
scale.  A threshold sweep (0.3 .. 0.9, three scales) found no operating point
either: separation stays between -34 and +1.3 points, and the only positive values
appear above 0.6-0.9 where the accept rate is under 1% (useless).  This is the same
domain shift Step 1 diagnosed - in this lighting the verifier's median P(hand) is
0.128 even on *correct* hand crops - now measured on blob crops.

## Validation (Steps 3.1-3.3)

**VeryLowLight, pipeline level** (`eval_vll_pipeline.py --motion-blobs`,
held-out split blocks marked):

| config | accepted | on the hand | clutter locks | held-out on-hand |
|---|---|---|---|---|
| v6/v7 (blobs OFF), default verifier | 130 (20.0%) | 32 (4.9%) | 98 | 7 / 200 |
| v6/v7 (blobs OFF), retrained verifier | 177 (27.2%) | 66 (10.1%) | 111 | 8 / 200 |
| blobs ON, default verifier | 455 (69.9%) | 50 (7.7%) | **405** | 10 / 200 |
| blobs ON, retrained verifier | 412 (63.3%) | 73 (11.2%) | **339** | 13 / 200 |

It does add on-hand positions in absolute terms (+18 all-frames with the default
verifier, +3 held-out; +7 and +5 with the retrained one) - but precision halves
(24.6% -> 11.0% of accepted positions are on the hand) and clutter locks quadruple
(98 -> 405).  **MediaPipe on the same clip: 44.7% static / 60.8% tracking**, so we
still do not beat it either way.

**7-clip regression** (`eval_motion_gate.py --motion-blobs --match-r 0.12`): with
blobs ON the numbers are *identical* to v6 - 469 hand frames / 15 clutter locks,
and **NohandLight / NohandDark stay at 0 acquisitions and 0 accepted frames**.  So
no regression there.  But this is not evidence of safety: those rooms are static
(no blobs exist to propose) and the daylight hand is found by the detector anyway,
so the new path simply never fires.  The danger only appears with *moving* clutter.

**Moving-clutter failure mode** (`test_moving_clutter.py`): none of our recorded
clips contain moving clutter, so it was constructed - `NohandLight.mp4` (a real
static room, no hand) with a textured patch cut from the frame itself translated
across the scene at 8 px/frame.  Every accepted position there is a false lock:

| config | accepted | on the moving (non-hand) object |
|---|---|---|
| blobs OFF (v6/v7) | 0 | - |
| blobs ON | 7 | **7 (100%)** |

With blobs on, the pipeline locks directly onto the moving non-hand object, every
time it locks at all.  Caveat: the object is a synthetic
translating patch, not a real curtain/fan/person, and no recorded clip contains
real moving clutter - so that specific case remains untested on real footage.

## Second look

* **Does it fix the localisation problem, or add a second, differently-flawed
  candidate source?**  Both, and the second dominates.  Motion locates the hand
  slightly more often than the detector does (42.2% vs 35.7% of hand frames), so
  the proposal is a genuine improvement.  But the confirmation gate does not
  discriminate in this lighting, so the accepted-position precision halves and
  false locks quadruple.  The detector's flaw is *failing to propose*; this path's
  flaw is *proposing and confirming the wrong thing*, which is worse - it creates
  locks that the detector-only pipeline never created (0 -> 7 on the injected
  moving object).
* **Does it generalise?**  The one clip again, one room, one lamp, one camera.
  Same limitation as the v7 retrain: measured on the target condition, not across
  conditions.
* **Regression risk in REACQUIRE, with numbers:** on the 7 existing clips, none
  (identical 469/15; both no-hand clips stay at 0 acquisitions, 0 accepted
  frames).  On moving clutter, severe: 100% lock rate onto a moving non-hand
  object, versus 0 for the current pipeline.

## Recommendation

**Keep it as an A/B option, not the default.**  The proposal idea is worth
pursuing, but it cannot ship on the strength of its own gating: "motion proposes,
verifier confirms" needs a verifier that works in the target lighting, and
`verylowlight_diagnosis.md` shows it does not.  The ordered fix is the one already
identified: fix detector localisation, or give the verifier data from this
condition *and* from moving non-hand objects, before trusting motion proposals.

`--motion-blobs` is available for live A/B, and the app prints a warning on
startup naming the 88-94% off-hand acceptance when it is enabled.
