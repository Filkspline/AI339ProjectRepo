"""Prove the v4 refactor did not change v10's behaviour.

`_on_hit` was restructured so it can WITHHOLD a position, and `self.pos` moved from
the top of the function down to the publish points.  With both consensus flags off
-- v10, and every build before the dark variant -- every accepted hit must still be
published and the rest of the state machine must reach exactly the same state as
before, frame for frame.

That claim is checked here by re-implementing the PRE-REFACTOR logic verbatim
(`ReferenceLocator`, copied from the version before this change) and driving both it
and the real HandLocator with identical random input sequences -- hits, misses, fast
jumps, repeats, and long runs -- asserting after EVERY step that every field matches
and that the real one published the hit.  Random input, fixed seed, so a failure is
reproducible.

Run: .venv-blazepalm\\Scripts\\python.exe test_v10_equivalence.py
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

from hand_mouse_cursor import (HandLocator, PlausibilityGate, MotionGate,  # noqa
                               REACQUIRE, TRACK, REACQUIRE_HITS, SEARCH_MARGIN,
                               MIN_WINDOW_PX, SHRINK_PER_HIT, WIDEN_PER_MISS,
                               MAX_WINDOW_FRAC, TRACK_LOST_TIME)

FIELDS = ("mode", "pos", "box_w", "window", "misses", "miss_time",
          "reacquire_hits")


class ReferenceLocator:
    """The pre-refactor state machine, verbatim (flags-off path)."""

    def __init__(self, margin=SEARCH_MARGIN, lost_time=TRACK_LOST_TIME):
        self.margin = float(margin)
        self.lost_time = float(lost_time)
        self.reset()

    def reset(self):
        self.mode = REACQUIRE
        self.pos = None
        self.box_w = None
        self.window = None
        self.misses = 0
        self.miss_time = 0.0
        self.reacquire_hits = 0

    def on_hit(self, pos, box_w):
        # ---- exactly what _on_hit used to do -------------------------------
        self.pos = pos
        self.box_w = box_w
        self.misses = 0
        self.miss_time = 0.0
        if self.mode == REACQUIRE:
            self.reacquire_hits += 1
            if self.reacquire_hits < REACQUIRE_HITS:
                return
            self.window = self.margin * box_w
            self.mode = TRACK
            self.reacquire_hits = 0
        else:
            self.window = max(self.margin * box_w,
                              MIN_WINDOW_PX,
                              self.window * SHRINK_PER_HIT)

    def on_miss(self, dt, fw, fh):
        self.misses += 1
        self.reacquire_hits = 0
        if self.mode != TRACK:
            return
        self.miss_time += max(float(dt), 0.0)
        if self.window is not None:
            self.window = min(self.window * WIDEN_PER_MISS,
                              MAX_WINDOW_FRAC * min(fw, fh))
        if self.miss_time >= self.lost_time:
            self.reset()


def same(a, b):
    """Field-by-field comparison; numpy-aware, tolerant of float noise."""
    out = []
    for f in FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if f == "pos":
            if (va is None) != (vb is None):
                out.append((f, va, vb))
            elif va is not None and not np.allclose(va, vb):
                out.append((f, va, vb))
        elif f == "window":
            if (va is None) != (vb is None):
                out.append((f, va, vb))
            elif va is not None and abs(float(va) - float(vb)) > 1e-9:
                out.append((f, va, vb))
        elif isinstance(va, float):
            if abs(va - vb) > 1e-9:
                out.append((f, va, vb))
        elif va != vb:
            out.append((f, va, vb))
    return out


def run_case(seed, n_steps, fw=640.0, fh=480.0):
    rng = random.Random(seed)
    real = HandLocator(None, gate=PlausibilityGate(), motion_gate=MotionGate())
    # flags off = v10 behaviour; assert that explicitly rather than assuming
    assert real.acquire_consensus == 0 and real.track_consensus == 0
    ref = ReferenceLocator()
    x, y = 300.0, 240.0
    dt = 1.0 / 30.0
    problems = []
    for step in range(n_steps):
        kind = rng.random()
        if kind < 0.12:
            real._on_miss(dt, fw, fh)
            ref.on_miss(dt, fw, fh)
        else:
            # mixture of small steps, big jumps, and repeats of the same place
            r = rng.random()
            if r < 0.55:
                x += rng.uniform(-20, 20)
                y += rng.uniform(-20, 20)
            elif r < 0.8:
                x += rng.uniform(-150, 150)
                y += rng.uniform(-120, 120)
            x = min(max(x, 20.0), fw - 20.0)
            y = min(max(y, 20.0), fh - 20.0)
            box_w = rng.uniform(40.0, 260.0)
            pos = np.array([x, y])
            published = real._on_hit(pos, box_w)
            if not published:
                problems.append(f"step {step}: flags-off hit was WITHHELD")
            ref.on_hit(pos, box_w)
        diff = same(real, ref)
        if diff:
            problems.append(f"step {step}: state diverged: {diff}")
            break
    return problems


def main():
    bad = 0
    cases = [(seed, 400) for seed in range(12)] + [(999, 1500)]
    for seed, n in cases:
        problems = run_case(seed, n)
        if problems:
            bad += 1
            print(f"[FAIL] seed {seed} ({n} steps)")
            for p in problems[:5]:
                print(f"        {p}")
        else:
            print(f"[PASS] seed {seed}: {n} steps, identical state and every "
                  f"accepted hit published")
    print(f"\n{len(cases)} random sequences: {len(cases) - bad} passed, {bad} failed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
