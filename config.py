"""Settings that produced the numbers in the report.

Everything that was tuned during the project ended up as a constant or a default
inside the scripts that use it. This file collects the values that matter for
reproducing the reported results in one place, so the README can point at it and a
reader can see the whole configuration without grepping.

The pipeline does not import this file. It is a reference table. If you want to
change a value for a run, use the matching command line flag, which is named in
the third column. The scripts' own defaults were not moved here, because they are
part of how the code was measured.
"""

# ---------------------------------------------------------------------------
# Which weights go with which result
# ---------------------------------------------------------------------------
# Report table / figure                     weights                        threshold
CONFIGS = {
    # The main comparison table (default configuration).
    "default": dict(verifier="verifier_cnn.pt", threshold=0.30, detector=None,
                    motion_blobs=False, acquire_consensus=0, track_consensus=0),
    # The low-light A/B configuration.
    "lowlight": dict(verifier="verifier_cnn_big.pt", threshold=0.61, detector=None,
                     motion_blobs=False, acquire_consensus=0, track_consensus=0),
    # The experimental dark build (HandCursorDark). These are the defaults of
    # hand_mouse_dark.py, which passes them to the app for you.
    "dark": dict(verifier="verifier_cnn_big.pt", threshold=0.61,
                 detector="palmdetector_dark.pth", motion_blobs=True,
                 acquire_consensus=3, acquire_consensus_tol=0.05,
                 track_consensus=3, track_consensus_tol=0.06),
    # The two ablations on the consensus work.
    "dark_acquire_only": dict(verifier="verifier_cnn_big.pt", threshold=0.61,
                              detector="palmdetector_dark.pth", motion_blobs=True,
                              acquire_consensus=3, acquire_consensus_tol=0.05,
                              track_consensus=0),
    "dark_blobs_off": dict(verifier="verifier_cnn_big.pt", threshold=0.61,
                           detector="palmdetector_dark.pth", motion_blobs=False,
                           acquire_consensus=0, track_consensus=0),
}

# ---------------------------------------------------------------------------
# Detection and acquisition
# ---------------------------------------------------------------------------
DETECTOR_SCORE_FLOOR = 0.5     # the app's floor; note the port itself drops
                               # detections below 0.7 inside its own NMS step
MIN_BOX_FRAC_REACQUIRE = 0.10  # full frame size gate, with a motion gate present
MIN_BOX_FRAC_REACQUIRE_NO_MOTION = 0.15   # the pre-v6 value, used with --no-motion-gate
MIN_BOX_FRAC_TRACK = 0.04      # in-window size gate, only used with --no-verifier
REACQUIRE_HITS = 2             # consecutive detections needed to commit a lock

# Motion gate (acquisition only)
MOTION_RING = 5                # frames in the differencing ring, about 170 ms at 30 fps
MOTION_WIDTH = 320             # the ring is a 320-wide INTER_AREA downscale
MOTION_PIXEL_THR = 12.0        # per-pixel luma change counted as changed, of 255
MOTION_MIN = 0.01              # minimum changed-pixel fraction in the candidate box
MOTION_BG_RATIO = 3.0          # candidate must also beat the median background tile
MOTION_BG_GRID = 4             # background tile grid is 4x4
GLOBAL_CHANGE_MAX = 3.0        # frame mean luma jump that suspends acquisition

# Motion-blob proposals (off by default, on in the dark build)
MOTION_BLOB_PIXEL_THR = 6.0
MOTION_BLOB_MIN_SIDE_FRAC = 0.05
MOTION_BLOB_MAX = 3
MOTION_BLOB_ROI_SCALE = 1.2    # set from measured geometry, not from performance
MOTION_BLOB_ROTATIONS = 8
MOTION_BLOB_PREFER_MARGIN = 0.10
MOTION_BLOB_THRESHOLD = 0.80   # threshold used with blob_verifier.pt

# ---------------------------------------------------------------------------
# Multi-pass agreement (dark build only, off everywhere else)
# ---------------------------------------------------------------------------
ACQUIRE_CONSENSUS_N = 3
ACQUIRE_CONSENSUS_TOL = 0.05   # fraction of frame width, about 32 px at 640 wide
TRACK_CONSENSUS_N = 3
TRACK_CONSENSUS_TOL = 0.06     # about 38 px at 640 wide
TRACK_CONSENSUS_MAX_HOLD = 6   # frames; bounds the hold so a fast move is not frozen out

# ---------------------------------------------------------------------------
# Tracking, smoothing and the cursor
# ---------------------------------------------------------------------------
SEARCH_MARGIN = 3.0            # window side = 3x the last known box width
WIDEN_PER_MISS = 1.4
SHRINK_PER_HIT = 1.0           # 1.0 means snap back to the base window
MIN_WINDOW_PX = 32.0
TRACK_LOST_TIME = 3.0          # seconds of accumulated misses before REACQUIRE
MAX_HAND_SPEED = 3.0           # frame widths per second, the plausibility limit
SMOOTHING_TAU = 0.35           # seconds, exponential moving average on screen pixels
MAX_SPEED = 1.0                # cursor cap, screen widths per second
LOSS_TIMEOUT = 3.0             # seconds with no position before the cursor lock drops

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Verifier (train_cnn_verifier.py, train_vll_verifier.py, blob_verifier.py)
VERIFIER_LR = 2e-3
VERIFIER_BATCH = 64            # blob_verifier.py uses 128
VERIFIER_EPOCHS = 8
VERIFIER_SEED = 0
VERIFIER_INPUT = 64            # network input; crops are stored at 128
VERIFIER_PARAMS = 60929        # SmallCNN, counted from the class definition

# Detector fine-tune (finetune_detector_dark.py)
FINETUNE_LR = 3e-4
FINETUNE_BATCH = 4
FINETUNE_EPOCHS = 20
FINETUNE_POS_WEIGHT = 5.0
FINETUNE_REG_WEIGHT = 0.5
FINETUNE_MAX_ANCHORS = 6       # anchors matched per ground-truth box
FINETUNE_LIGHT_FRAC = 0.25     # fraction of samples left un-augmented
COLLAPSE_GUARD_HOT = 100       # refuse to save above this many anchors >= 0.5
DETECTOR_PARAMS = 1763358
DETECTOR_HEAD_PARAMS = 43966   # the six 1x1 heads, 2.49% of the detector

# Logistic baseline (train_verifier.py)
LOGISTIC_LR = 0.05
LOGISTIC_WEIGHT_DECAY = 1e-3
LOGISTIC_EPOCHS = 3000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Ground truth for the recorded clips is the lag-free static-mode MediaPipe run.
TRUTH_MODE = "mediapipe static, per frame, gaps borrowed within +-5 frames"
MATCH_RADIUS = "max(0.06 frame widths, 0.6 x palm width)"
BLOCK, HOLDOUT_EVERY = 50, 3   # every third block of 50 frames is held out

# Augmentation matches the target clip's measured photometric statistics.
TARGET_CLIP = "VeryLowLight.mp4"
TARGET_MEAN_BGR = (32.9, 38.7, 57.4)
TARGET_STD_BGR = (19.3, 24.0, 34.0)
AUG_SEVERITIES_BIG = (0.5, 0.8, 1.0)
AUG_SEVERITIES_DARK = (0.35, 0.55, 0.75, 1.0)
AUG_DOWNSCALE = (0.2, 1.0)
ARBITER_MIN_PRESENCE = 0.5     # landmark presence floor for a mined crop to be kept
HAGRID_NO_GESTURE_LABEL = 13   # class index that is skipped during mining

# ---------------------------------------------------------------------------
# Clip library. Also listed in evaluate_pipelines.py, which reads the files from
# the working directory.
# ---------------------------------------------------------------------------
CLIPS = {
    "CloseLightMoving.mp4": "hand, close, daylight, moving",
    "CloseLightStill.mp4": "hand, close, daylight, held still",
    "CloseDarkStill.mp4": "hand, close, dark, held still",
    "FarLight.mp4": "hand, far, daylight",
    "FarDark.mp4": "hand, far, dark",
    "NohandLight.mp4": "no hand, static room, daylight",
    "NohandDark.mp4": "no hand, static room, dark",
    "VeryLowLight.mp4": "hand, close, artificial light in a slightly dark room",
}
