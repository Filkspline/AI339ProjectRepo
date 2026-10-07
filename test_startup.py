"""Startup and loop smoke tests, in every configuration AND in the frozen builds.

This file exists because two real bugs shipped past manual checks:

  1. `HandCursor_v9` -- `blob_verifier` was only assigned inside
     `if args.motion_blobs:`, so the app raised UnboundLocalError at startup in its
     DEFAULT configuration.  Every smoke test that round used `--motion-blobs`.
  2. `HandCursorDark_v1` -- `--verifier verifier_cnn_big.pt` was resolved relative to
     the process cwd only.  A double-click sets cwd to the exe's own folder, so the
     verifier silently went inactive, and the motion-blob path then called a None
     verifier and died about a second after the camera opened.  The sanity check had
     run the exe from the project root (where the file IS reachable) and on a machine
     with no webcam (so the loop never executed).  Both gaps are closed here: the
     frozen builds are launched with cwd set to their own directory, and the locator
     loop is exercised headlessly over a real clip.

Checks:
  A. every source-level flag combination starts and reaches the webcam step
  B. the locator loop runs over real frames in the configurations that crashed
  C. a bare model filename resolves from a DIFFERENT working directory
  D. each frozen build in dist/ starts with ZERO arguments and cwd = its own
     directory (the true double-click path), no traceback, and reports its verifier
     as loaded when one is expected

Run: .venv-blazepalm\\Scripts\\python.exe test_startup.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "blazepalm")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)
# The original project ran this from a venv sitting next to the code. On a fresh
# clone that venv does not exist, so fall back to whichever interpreter is running
# this file.
_VENV_PY = os.path.join(HERE, ".venv-blazepalm", "Scripts", "python.exe")
PY = _VENV_PY if os.path.isfile(_VENV_PY) else sys.executable
ENTRY = os.path.join(HERE, "hand_mouse_cursor.py")
DARK_ENTRY = os.path.join(HERE, "hand_mouse_dark.py")

# (build directory name, must its zero-arg log show a loaded verifier?,
#  known_broken, expected acquire-consensus state, expected TRACK-hold state)
# HandCursorDark_v1 is kept as a CANARY: it must still fail the
# "verifier loaded" bar.  If that build ever starts passing, this test has lost its
# teeth (or someone edited a shipped exe) and the canary reports FAIL.
# Consensus state: "absent" = the build predates the feature, so its startup output
# must NOT mention it at all; 0 = must report OFF; N>1 = must report that N.
# This is the wiring check for the shipped artifact: a wrapper that failed to inject
# a flag would otherwise be invisible, because the app starts fine either way.
FROZEN = [("HandCursor_v10", True, False, "absent", "absent"),   # pre-consensus
          ("HandCursorDark_v1", True, True, "absent", "absent"),  # broken control
          ("HandCursorDark_v2", True, False, "absent", "absent"),  # pre-consensus
          ("HandCursorDark_v3", True, False, 3, "absent"),   # acquire only
          ("HandCursorDark_v4", True, False, 3, 3)]          # acquire + TRACK hold

SOURCE_CONFIGS = [
    ("default (double-click path)", ENTRY, []),
    ("--motion-blobs", ENTRY, ["--motion-blobs"]),
    ("--motion-blobs --no-blob-verifier", ENTRY,
     ["--motion-blobs", "--no-blob-verifier"]),
    ("--no-motion-gate", ENTRY, ["--no-motion-gate"]),
    ("--no-verifier", ENTRY, ["--no-verifier"]),
    ("--verifier-threshold 0.6", ENTRY, ["--verifier-threshold", "0.6"]),
    ("--verifier verifier_cnn_big.pt", ENTRY,
     ["--verifier", "verifier_cnn_big.pt"]),
    ("--detector palmdetector_dark.pth", ENTRY,
     ["--detector", "palmdetector_dark.pth"]),
    ("dark variant (zero args)", DARK_ENTRY, []),
    ("dark variant --no-motion-blobs", DARK_ENTRY, ["--no-motion-blobs"]),
    ("dark variant --verifier verifier_dark.pt", DARK_ENTRY,
     ["--verifier", "verifier_dark.pt"]),
    ("dark variant consensus off (--acquire-consensus 0)", DARK_ENTRY,
     ["--acquire-consensus", "0"]),
    ("dark variant TRACK hold off (--track-consensus 0, v3 behaviour)", DARK_ENTRY,
     ["--track-consensus", "0"]),
    ("--acquire-consensus 3 (main app, flag must work there too)", ENTRY,
     ["--acquire-consensus", "3"]),
    ("--acquire-consensus 3 --acquire-consensus-tol 0.08", ENTRY,
     ["--acquire-consensus", "3", "--acquire-consensus-tol", "0.08"]),
    ("--track-consensus 3 (main app)", ENTRY, ["--track-consensus", "3"]),
    ("--track-consensus 2 --track-consensus-tol 0.12", ENTRY,
     ["--track-consensus", "2", "--track-consensus-tol", "0.12"]),
]


def run_capture(cmd, cwd):
    out = os.path.join(tempfile.gettempdir(), "dsh_startup_smoke.txt")
    with open(out, "w", encoding="utf-8", errors="replace") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, timeout=600)
    return open(out, encoding="utf-8", errors="replace").read()


def check_source_configs():
    res = []
    for name, entry, args in SOURCE_CONFIGS:
        txt = run_capture([PY, entry, *args], HERE)
        reached = "could not open webcam" in txt or "Models loaded" in txt
        head = txt.split("could not open webcam")[0]
        crashed = "Traceback" in head
        detail = "reached webcam step" if reached else "did NOT reach webcam step"
        if crashed:
            last = [l for l in head.splitlines() if l.strip()]
            detail = f"traceback before startup: {last[-1][:80] if last else '?'}"
        res.append((f"source: {name}", reached and not crashed, detail))
    return res


def check_loop():
    """Headless: the locator loop over real frames, in the configurations that
    crashed.  No webcam needed -- which is the whole point, since the crash happened
    inside the loop and every webcam-less startup check missed it."""
    import cv2
    from blazepalm import PalmDetector
    from hand_mouse_cursor import (HandLocator, PlausibilityGate, MotionGate,
                                   load_blob_verifier, load_verifier)

    det = PalmDetector()
    det.load_weights(os.path.join(ML, "palmdetector.pth"))
    det.load_anchors(os.path.join(ML, "anchors.npy"))
    det.eval()
    ver, _p = load_verifier("verifier_cnn_big.pt")
    blob_ver, _bp = load_blob_verifier(None)
    frames = []
    cap = cv2.VideoCapture(os.path.join(HERE, "CloseLightMoving.mp4"))
    for _ in range(40):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    cases = [
        ("blobs on, verifier None (the crash case)",
         dict(verifier=None, use_motion_blobs=True, blob_verifier=blob_ver)),
        ("blobs on, verifier loaded",
         dict(verifier=ver, use_motion_blobs=True, blob_verifier=blob_ver)),
        ("blobs on, nothing loaded at all",
         dict(verifier=None, use_motion_blobs=True, blob_verifier=None)),
        ("blobs off, verifier loaded", dict(verifier=ver, use_motion_blobs=False)),
    ]
    res = []
    for name, kw in cases:
        try:
            loc = HandLocator(det, gate=PlausibilityGate(), motion_gate=MotionGate(),
                              verifier_threshold=0.61, **kw)
            t = 0.0
            for fr in frames:
                t += 1.0 / 30.0
                loc.update(fr, t, 1.0 / 30.0)
            res.append((f"loop: {name}", True,
                        f"survived {len(frames)} frames, no exception"))
        except Exception as exc:                              # noqa: BLE001
            res.append((f"loop: {name}", False, f"{type(exc).__name__}: {exc}"))
    return res


def check_cwd_independence():
    tmp = os.path.join(tempfile.gettempdir(), "dsh_cwd_check")
    os.makedirs(tmp, exist_ok=True)
    code = (
        "import os,sys; sys.path.insert(0, r'%s'); import hand_mouse_cursor as h;"
        "v,p = h.load_verifier('verifier_cnn_big.pt');"
        "d = h.resolve_model_path('palmdetector_dark.pth');"
        "b,_q = h.load_blob_verifier('blob_verifier.pt');"
        "print('CWD', os.getcwd());"
        "print('VERIFIER', 'LOADED' if v else 'NONE');"
        "print('DETECTOR', 'LOADED' if d else 'NONE');"
        "print('BLOBVER', 'LOADED' if b else 'NONE')" % HERE)
    txt = run_capture([PY, "-c", code], tmp)
    ok = ("VERIFIER LOADED" in txt and "DETECTOR LOADED" in txt
          and "BLOBVER LOADED" in txt)
    return [("bare model names resolve from another cwd", ok,
             txt.strip().replace("\n", " | ")[:140])]


def check_frozen():
    res = []
    for name, expect_verifier, known_broken, expect_consensus, expect_track in FROZEN:
        d = os.path.join(HERE, "dist", name)
        exe = os.path.join(d, f"{name}.exe")
        if not os.path.isfile(exe):
            res.append((f"frozen zero-arg cwd=exe: {name}", False, "not built"))
            continue
        # THE double-click emulation: cwd is the exe's own directory, zero args
        txt = run_capture([exe], d)
        reached = "could not open webcam" in txt
        crashed = "Traceback" in txt
        ver_loaded = "Verifier loaded" in txt
        healthy = reached and not crashed and (ver_loaded or not expect_verifier)
        detail = ("reached webcam step, verifier loaded" if ver_loaded
                  else "reached webcam step, verifier NOT loaded")
        if crashed:
            last = [l for l in txt.split("could not open webcam")[0].splitlines()
                    if l.strip()]
            detail = f"traceback: {last[-1][:80] if last else '?'}"
        elif not reached:
            detail = "did NOT reach webcam step"
        # the built artifact must actually be wired: its zero-arg startup line has
        # to name the consensus settings this version was built with
        if reached and not crashed:
            for want, label, off_text in ((expect_consensus, "acquire", "OFF"),
                                          (expect_track, "TRACK hold", "OFF")):
                key = "ACQUIRE consensus" if label == "acquire" else "TRACK consensus"
                if want == "absent":
                    if key in txt:
                        healthy = False
                        detail = (f"startup line mentions {label} consensus, but "
                                  f"this build predates the feature")
                    else:
                        detail += f", no {label} line"
                elif want == 0:
                    if f"{key} = OFF" not in txt:
                        healthy = False
                        detail = f"startup line does NOT report {label} consensus OFF"
                    else:
                        detail += f", {label} OFF"
                elif want:
                    if f"{key} = ON" not in txt or f"{want} consecutive" not in txt:
                        healthy = False
                        detail = (f"startup line does NOT report {label} consensus "
                                  f"{want} consecutive")
                    else:
                        detail += f", {label} {want}-pass"
        if known_broken:
            ok = not healthy
            detail = ("canary OK: this build is still detected as broken "
                      f"({detail})") if ok else \
                     "CANARY LOST: the known-broken build now looks healthy"
        else:
            ok = healthy
        res.append((f"frozen zero-arg cwd=exe: {name}", ok, detail))
    return res


def check_consensus_wiring():
    """The dark variant's zero-arg path must actually turn the consensuses ON, and
    the main app's must leave them OFF.  A wrapper that silently fails to inject the
    flags would otherwise be invisible: the app would start fine and just behave
    like v10."""
    res = []
    txt = run_capture([PY, DARK_ENTRY], HERE)
    on = ("ACQUIRE consensus = ON: 3 consecutive" in txt
          and "TRACK consensus = ON" in txt and "3 consecutive" in txt)
    lines = [l for l in txt.splitlines() if "consensus" in l]
    res.append(("dark zero-arg enables both consensuses", on, lines[:2]))
    txt = run_capture([PY, ENTRY], HERE)
    off = ("ACQUIRE consensus = OFF" in txt and "TRACK consensus = OFF" in txt
           and "consensus = ON" not in txt)
    lines = [l for l in txt.splitlines() if "consensus" in l]
    res.append(("main app zero-arg leaves both consensuses OFF", off, lines[:2]))
    return [(n, ok, " | ".join(map(str, d)) if isinstance(d, list) else d)
            for n, ok, d in res]


def main():
    results = []
    print("== A. source configurations (zero-arg path included) ==")
    results += check_source_configs()
    print("\n== B. locator loop over real frames (headless) ==")
    results += check_loop()
    print("\n== C. model paths independent of cwd ==")
    results += check_cwd_independence()
    # Section D needs the packaged builds in dist/. Those are compiled artefacts, not
    # source, so a fresh clone has none. build_release.py produces them.
    if os.path.isdir(os.path.join(HERE, "dist")):
        print("\n== D. frozen builds: zero arguments, cwd = the exe's directory ==")
        results += check_frozen()
    else:
        print("\n== D. frozen builds: skipped (no dist/ here; run build_release.py to make one) ==")
    print("\n== E. acquire-consensus wiring (dark on, main off) ==")
    results += check_consensus_wiring()

    print("\n== summary ==")
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")
    n = sum(1 for _n, ok, _d in results if ok)
    print(f"\n{len(results)} checks: {n} passed, {len(results) - n} failed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
