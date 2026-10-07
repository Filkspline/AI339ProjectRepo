"""Build a versioned, double-clickable release of hand_mouse_pytorch.py.

Standing process (see VERSION_LOG.md):
  * onedir by default (no per-launch extraction), onefile only on request
  * never overwrite an existing version -- dist/HandTrackerPyTorch_vN/
  * sanity-check the exe directly (no venv) before declaring it ready
  * append an entry to VERSION_LOG.md

Usage:
    python build_release.py --summary "v2 -- replaced box-size gate with crop classifier"
    python build_release.py --summary "..." --onefile      # convenience build only

Exits non-zero if the build or the sanity check fails.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")
LOG = os.path.join(HERE, "VERSION_LOG.md")
ENTRY = "hand_mouse_pytorch.py"
ML_REL = os.path.join("blazepalm")
NAME_BASE = "HandTrackerPyTorch"
# Entry points that are REQUIRED to load a verifier; used by the build-time sanity
# check to refuse a build whose verifier silently went inactive.
VERIFIER_ENTRIES = {"hand_mouse_cursor.py", "hand_mouse_dark.py"}
PYINSTALLER = os.path.join(HERE, ".venv-blazepalm", "Scripts", "pyinstaller.exe")
DATA_FILES = ["palmdetector.pth", "HandLandmarks.pth", "anchors.npy"]
# Files that live in the project root rather than next to the ML code (e.g. the
# verifiers trained by train_cnn_verifier.py / train_vll_verifier.py).  They are
# bundled into the same `models` folder so a frozen build finds them through
# hand_pipeline.ML_DIR, exactly like the detector weights.  Only included if
# present, so the landmark app still builds if they have not been trained yet.
# verifier_cnn_vll.pt is the dark/artificial-light-tuned model: NOT the default
# (it regresses the daylight clips -- see VLL_DIAGNOSIS.md), but bundled so it can
# be switched in live with --verifier verifier_cnn_vll.pt.
# verifier_cnn_big.pt is the 130k-crop / real-HaGRID-volume model: also NOT the
# default (3x the clutter locks on the validated clips), also bundled for live A/B
# with --verifier verifier_cnn_big.pt --verifier-threshold 0.61.  See VERIFIER_V2.md.
# verifier_dark.pt and palmdetector_dark.pth belong to the dark-specialised
# variant (HandCursorDark, entry hand_mouse_dark.py); the originals stay first so
# nothing about v10's defaults changes.
ROOT_DATA = ["verifier_cnn.pt", "verifier_cnn_vll.pt", "blob_verifier.pt",
             "verifier_cnn_big.pt", "verifier_dark.pt", "palmdetector_dark.pth"]
# Dataset-prep dependencies.  torch's PyInstaller hook pulls pyarrow in when it
# happens to be installed, which cost 79 MB of dead weight in HandCursor_v3
# (pyarrow is only used by build_verifier_dataset.py, never at runtime).
EXCLUDES = ["pyarrow"]


def next_version(name_base=NAME_BASE):
    if not os.path.exists(LOG):
        return 1
    versions = [int(v) for v in re.findall(rf"{name_base}_v(\d+)", open(LOG).read())]
    return max(versions) + 1 if versions else 1


def build(name, onefile, entry=ENTRY):
    mode = "--onefile" if onefile else "--onedir"
    cmd = [PYINSTALLER, mode, "--noconfirm", "--name", name,
           # --paths is REQUIRED: hand_pipeline/blazepalm/handlandmarks live in a
           # folder added to sys.path at RUNTIME, so PyInstaller's static
           # analysis misses them (and then misses torch) without this.
           "--paths", ML_REL]
    for f in DATA_FILES:
        cmd += ["--add-data", f"{os.path.join(ML_REL, f)};models"]
    for f in ROOT_DATA:
        if os.path.exists(os.path.join(HERE, f)):
            cmd += ["--add-data", f"{f};models"]
        else:
            print(f"   note: {f} not present -- not bundled")
    for m in EXCLUDES:
        cmd += ["--exclude-module", m]
    cmd.append(entry)
    print("  ", " ".join(cmd[:6]), "...")
    subprocess.run(cmd, cwd=HERE, check=True)


def exe_path(name, onefile):
    if onefile:
        return os.path.join(DIST, f"{name}.exe")
    return os.path.join(DIST, name, f"{name}.exe")


def sanity_check(exe, expect_verifier=False):
    """Run the exe directly (no venv) and confirm it reaches the webcam step.

    Output is redirected to a FILE rather than a pipe (the sandbox blocks some
    IPC), then the file is read back.

    The exe is launched with cwd set to its OWN DIRECTORY, because that is what a
    double-click does.  An earlier version ran it with cwd = the project root, and
    that is exactly the gap that let HandCursorDark_v1 ship broken: its
    `--verifier verifier_cnn_big.pt` was resolved against the cwd, so from the
    project root it loaded fine and from the exe's own folder it silently did not.
    """
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    out = os.path.join(HERE, "build", "sanity_output.txt")
    exe_dir = os.path.dirname(os.path.abspath(exe))
    with open(out, "w", encoding="utf-8", errors="replace") as f:
        subprocess.run([exe], stdout=f, stderr=subprocess.STDOUT, cwd=exe_dir,
                       timeout=600)
    txt = open(out, encoding="utf-8", errors="replace").read()
    reached_models = "Models loaded" in txt
    reached_webcam = "could not open webcam" in txt
    crashed = "Traceback" in txt
    verifier_loaded = "Verifier loaded" in txt
    print(f"   launched with cwd            : {exe_dir}")
    print(f"   reached 'Models loaded'      : {reached_models}")
    print(f"   reached webcam-open step     : {reached_webcam}")
    print(f"   verifier weights loaded      : {verifier_loaded}"
          + ("  (expected)" if expect_verifier else ""))
    print(f"   crashed with a traceback     : {crashed}")
    ok = reached_models and reached_webcam and not crashed
    if expect_verifier and not verifier_loaded:
        ok = False
        print("   FAILED: the verifier was expected to load but did not")
    if reached_webcam:
        # No camera on the build machine means only the startup path ran; the
        # locator loop -- where v9 and the dark v1 crash actually lived -- is
        # covered headlessly by `test_startup.py` section B.  Say so out loud
        # rather than implying the loop was exercised.
        print("   NOTE: no webcam here, so the interactive loop was NOT exercised "
              "by this check.")
        print("         Run test_startup.py (section B) for the headless loop test.")
    if not ok:
        print("   --- output was ---")
        print(txt)
    return ok


def log_entry(version, name, onefile, size_mb, summary, ok):
    new = not os.path.exists(LOG)
    if new:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("# Build version log\n\n")
            f.write("Versioned double-click builds of `hand_mouse_pytorch.py`. "
                    "Never overwritten -- older versions stay in `dist/` for "
                    "rollback/comparison.\n\n")
            f.write("| version | date | mode | size | summary | sanity |\n")
            f.write("|---|---|---|---|---|---|\n")
    date = datetime.date.today().isoformat()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"| `{name}` | {date} | {'onefile' if onefile else 'onedir'} | "
                f"{size_mb:.0f} MB | {summary} | "
                f"{'OK (reached webcam step)' if ok else '**FAILED**'} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="one-line description of what changed in this build")
    ap.add_argument("--version", type=int, default=None,
                    help="override the auto-incremented version number")
    ap.add_argument("--onefile", action="store_true",
                    help="convenience single-file build (not the default)")
    ap.add_argument("--entry", default=ENTRY,
                    help=f"script to package (default {ENTRY})")
    ap.add_argument("--name-base", default=NAME_BASE,
                    help="build name / version sequence (default "
                         f"{NAME_BASE}); each program keeps its own sequence")
    args = ap.parse_args()

    if not os.path.exists(PYINSTALLER):
        raise SystemExit(f"pyinstaller not found at {PYINSTALLER}")
    if not os.path.exists(os.path.join(HERE, args.entry)):
        raise SystemExit(f"entry script not found: {args.entry}")

    v = args.version or next_version(args.name_base)
    name = f"{args.name_base}_v{v}"
    exe = exe_path(name, args.onefile)
    if os.path.exists(exe):
        raise SystemExit(f"refusing to overwrite existing build: {exe}")

    print(f"== building {name} ({'onefile' if args.onefile else 'onedir'}) "
          f"from {args.entry} ==")
    build(name, args.onefile, args.entry)

    if not os.path.exists(exe):
        raise SystemExit(f"build finished but {exe} is missing")
    if args.onefile:
        size_mb = os.path.getsize(exe) / (1024 ** 2)
    else:
        size_mb = sum(os.path.getsize(os.path.join(r, f))
                      for r, _, fs in os.walk(os.path.join(DIST, name))
                      for f in fs) / (1024 ** 2)

    print(f"== sanity check: running {exe} directly (no venv) ==")
    # Both entries are REQUIRED to end up with a working verifier: the main app
    # loads the bundled one by default, and hand_mouse_dark.py asks for
    # verifier_cnn_big.pt explicitly.  The old rule keyed only on the main entry,
    # so the dark build passed its build-time check with the verifier inactive --
    # the exact failure mode that shipped in HandCursorDark_v1.
    expect_verifier = os.path.basename(args.entry) in VERIFIER_ENTRIES
    ok = sanity_check(exe, expect_verifier=expect_verifier)

    log_entry(v, name, args.onefile, size_mb, args.summary, ok)
    print(f"\n  exe  : {exe}")
    print(f"  size : {size_mb:.0f} MB")
    print(f"  log  : {LOG}")
    if not ok:
        print("\nBUILD NOT READY: sanity check failed (see output above).")
        return 1
    print(f"\n{name} is ready to test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
