"""HandCursorDark: the dark-specialised variant entry point.

SEPARATE from HandCursor_v10 on purpose.  v10 stays the safe general-purpose
default, untouched; this build is allowed to trade light-condition precision for
dark-condition recall, and is labelled experimental everywhere it is logged.

What it changes, relative to v10's defaults:

  --detector palmdetector_dark.pth
      BlazePalm's heads (2.49% of parameters) fine-tuned on VeryLowLight train
      blocks with the dark/distance augmentation.  The original weights on disk are
      untouched.  Measured detector-only localisation: VeryLowLight holdout 25.0%
      -> 33.0%, with the light clips essentially neutral overall (+0.4 points mean
      across seven unseen clips, gaining on the far ones, losing ~6-10 points on
      the already-saturated close ones).

  --verifier verifier_cnn_big.pt --verifier-threshold 0.61
      Chosen by measurement over the Step-1 retrain.  A dark/distance-heavy
      augmentation of the same crop pool was trained (verifier_dark.pt) and it does
      beat the 11k baseline handsomely (VeryLowLight holdout hand crops kept
      22.9% -> 55.4%), but at MATCHED own-room precision it still loses to the
      130k-crop volume model: 47.0% vs 57.8% of MediaPipe-anchored hand crops kept
      (46.3% vs 58.5% on detector-recipe crops).  So the dark variant ships the
      model that actually measures best for dark, and verifier_dark.pt stays
      available as an alternative (--verifier verifier_dark.pt
      --verifier-threshold 0.70).

  --motion-blobs
      Motion proposals ON by default here (Step 3): in very low light a user will
      naturally wave, so leaning on motion is a deliberate trade for this variant.
      Still confirmed by the blob-specific verifier, which is the part that keeps
      moving clutter out.

  --acquire-consensus 3 --acquire-consensus-tol 0.05
      Multi-pass spatial consensus on REACQUIRE (v3): a lock is only committed
      after three consecutive accepted passes that all sit within 0.05 x frame
      width of each other, so a detector flickering between the hand and something
      else cannot commit.  Costs acquisition latency on the intermittent clips --
      measured, and reported in DARK_CONSENSUS.md.  TRACK is untouched.

Any flag given on the command line wins over these defaults.

Usage:
    .venv-blazepalm\\Scripts\\python.exe hand_mouse_dark.py            # dark variant
    .venv-blazepalm\\Scripts\\python.exe hand_mouse_dark.py --no-motion-blobs
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DARK_VERIFIER_THRESHOLD = 0.61      # picked in sweep_verifier_threshold.py
# Multi-pass spatial consensus (v3, Step 1 of the "stop jumping to the armpit"
# round): the fine-tuned detector flickers between the hand and nearby clutter in
# dim light, so no single frame is trusted.  Chosen from measurement, not taste --
# see DARK_CONSENSUS.md and the constants block in hand_mouse_cursor.py.
DARK_CONSENSUS_N = 3
DARK_CONSENSUS_TOL = "0.05"
# TRACK side of the same idea (v4): an in-window candidate that jumps off the hand
# must not move the cursor until it repeats.  This is where the off-hand frames
# actually come from -- 307 of 317 -- so it is the change that addresses the live
# "jumps to an armpit" symptom rather than just acquisition.
DARK_TRACK_CONSENSUS_N = 3
DARK_TRACK_CONSENSUS_TOL = "0.06"
DARK_DEFAULTS = ["--detector", "palmdetector_dark.pth",
                 "--verifier", "verifier_cnn_big.pt",
                 "--verifier-threshold", str(DARK_VERIFIER_THRESHOLD),
                 "--acquire-consensus", str(DARK_CONSENSUS_N),
                 "--acquire-consensus-tol", DARK_CONSENSUS_TOL,
                 "--track-consensus", str(DARK_TRACK_CONSENSUS_N),
                 "--track-consensus-tol", DARK_TRACK_CONSENSUS_TOL,
                 "--motion-blobs"]


def main():
    argv = sys.argv[1:]
    given = {a.split("=")[0] for a in argv}
    for i in range(0, len(DARK_DEFAULTS), 2):
        flag = DARK_DEFAULTS[i]
        if flag not in given:
            argv += DARK_DEFAULTS[i:i + 2]
    if ("--motion-blobs" not in given and "--no-motion-blobs" not in given
            and "--motion-blobs" not in argv):
        argv.append("--motion-blobs")
    sys.argv = [sys.argv[0]] + argv
    print("HandCursorDark: dark-specialised variant "
          "(fine-tuned detector + dark verifier + motion blobs on + "
          f"{DARK_CONSENSUS_N}-pass spatial consensus on REACQUIRE + "
          f"{DARK_TRACK_CONSENSUS_N}-pass confirmation before TRACK moves the "
          f"cursor)")
    import hand_mouse_cursor as hmc
    hmc.main()


if __name__ == "__main__":
    main()
