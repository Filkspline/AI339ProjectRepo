"""Synthetic tests for CursorController (no video, no webcam).

Verifies the properties that matter for a predictable cursor:
  1. a verified position locks and drives the cursor on the FIRST frame (the
     acquisition dwell was removed -- there is no hover-to-lock any more);
  2. the cursor never moves faster than the speed cap, acquisition included;
  3. it holds EXACTLY still during detection gaps and keeps the lock;
  4. a single noisy spike barely moves it (because it is rate limited);
  5. loss only drops the lock after the full timeout, and re-acquiring after a
     loss is again immediate.

Run: .venv-blazepalm\\Scripts\\python.exe test_cursor_controller.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_mouse_cursor import CursorController, SEARCHING, LOCKED  # noqa: E402

DT = 1.0 / 30.0
SW, SH = 1920, 1080


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if detail:
        print(f"        {detail}")
    return bool(cond)


def lock(ctrl, pos=None, dt=DT):
    """Feed one verified position; the lock must be immediate."""
    p = ctrl.cursor.copy() if pos is None else np.asarray(pos, float)
    _, s = ctrl.update(p, dt)
    return s == LOCKED


def main():
    r = []

    # ---- acquisition is immediate on a verified position -----------------
    c = CursorController(SW, SH)
    one = lock(c)
    r.append(check("a verified position locks on the FIRST frame (no dwell)",
                   one, f"state={c.state} after one frame"))

    c = CursorController(SW, SH)
    c.update(None, DT)
    c.update(None, DT)
    r.append(check("no position -> never locks",
                   c.state == SEARCHING, f"state={c.state}"))

    # a far-away verified position must not snap the cursor there
    c = CursorController(SW, SH, max_speed=1.0)
    start = c.cursor.copy()
    cap = c.max_speed_px * DT
    lock(c, start + np.array([900.0, 0.0]))
    step = float(np.linalg.norm(c.cursor - start))
    r.append(check("first lock after acquisition does NOT snap the cursor",
                   c.state == LOCKED and step <= cap + 1e-6,
                   f"first step {step:.3f} px vs cap {cap:.3f} px/frame"))

    # ---- cursor does not move while searching ---------------------------
    c = CursorController(SW, SH)
    before = c.cursor.copy()
    for _ in range(20):
        c.update(None, DT)                             # no verified position
    r.append(check("cursor stays put while SEARCHING",
                   np.allclose(c.cursor, before) and c.state == SEARCHING,
                   f"moved {np.linalg.norm(c.cursor - before):.3f} px"))

    # ---- speed cap ------------------------------------------------------
    c = CursorController(SW, SH, max_speed=1.0)
    lock(c)
    far = c.cursor + np.array([900.0, 0.0])
    worst = 0.0
    for _ in range(60):
        b = c.cursor.copy()
        c.update(far, DT)
        worst = max(worst, float(np.linalg.norm(c.cursor - b)))
    r.append(check("cursor never exceeds the speed cap",
                   worst <= cap + 1e-6,
                   f"largest step {worst:.3f} px vs cap {cap:.3f} px/frame"))

    # ---- hold exactly still during gaps --------------------------------
    c = CursorController(SW, SH)
    lock(c)
    held = c.cursor.copy()
    for _ in range(10):
        c.update(None, DT)
    r.append(check("cursor holds EXACTLY still during a detection gap",
                   np.array_equal(c.cursor, held) and c.state == LOCKED,
                   f"delta={np.linalg.norm(c.cursor - held):.6f} px, state={c.state}"))

    # ---- a single noisy spike cannot drag the cursor --------------------
    c = CursorController(SW, SH, max_speed=1.0)
    lock(c)
    start = c.cursor.copy()
    c.update(start + np.array([900.0, 0.0]), DT)       # one-frame spike
    c.update(start, DT)                                # back to normal
    net = float(np.linalg.norm(c.cursor - start))
    r.append(check("single-frame spike barely moves the cursor",
                   net <= 2 * cap + 1e-6,
                   f"net displacement {net:.3f} px (spike was 900 px)"))

    # ---- sustained far target still obeys the cap (no snap on resume) ---
    c = CursorController(SW, SH, max_speed=1.0)
    lock(c)
    c.update(None, DT)                                 # brief gap
    b = c.cursor.copy()
    c.update(b + np.array([500.0, 300.0]), DT)         # resumes far away
    r.append(check("resume after a gap does not snap",
                   np.linalg.norm(c.cursor - b) <= cap + 1e-6,
                   f"first step after gap {np.linalg.norm(c.cursor - b):.3f} px"))

    # ---- loss timeout, and immediate re-acquisition ---------------------
    c = CursorController(SW, SH, loss_timeout=1.0)
    lock(c)
    for _ in range(int(0.5 / DT)):                     # 0.5 s < timeout
        c.update(None, DT)
    still_locked = c.state == LOCKED
    for _ in range(int(0.7 / DT)):                     # now past 1.0 s total
        c.update(None, DT)
    r.append(check("lock survives short loss, drops after the full timeout",
                   still_locked and c.state == SEARCHING,
                   f"after 0.5s={LOCKED if still_locked else c.state}, "
                   f"after 1.2s={c.state}"))

    reacquired = lock(c)
    r.append(check("re-acquisition after a loss is immediate again",
                   reacquired, f"state={c.state}"))

    n_pass = sum(r)
    print(f"\n{len(r)} tests: {n_pass} passed, {len(r) - n_pass} failed")
    return 0 if n_pass == len(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
