"""The VeryLowLight train / validation / test split, and the rule that defines it.

Blocks of 50 frames are dealt to the three roles by a seeded shuffle. Every decision
that touches VeryLowLight (verifier threshold, augmentation recipe, detector epochs,
agreement tolerances) is made on the train and validation blocks. The test blocks are
evaluated once, at the end, and never used to choose anything.

Recorded so the split is reproducible: the seed and the block assignment are written to
results/vll_split.json.

    python vll_split.py            # prints the assignment and writes the json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "vll_split.json")

BLOCK = 50
SPLIT_SEED = 0
CLIP_FRAMES = 651
# five train blocks, four validation, four test. Test is 200 frames, which is the
# resolution the old held-out scheme had, and enough for the hand labelling subset.
N_TRAIN, N_VAL, N_TEST = 5, 4, 4


def block_roles(seed=SPLIT_SEED, n_frames=CLIP_FRAMES):
    """Return {block_index: role} for the seeded deal."""
    n_blocks = (n_frames + BLOCK - 1) // BLOCK
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    roles = {}
    for i, b in enumerate(order):
        if i < N_TRAIN:
            roles[int(b)] = "train"
        elif i < N_TRAIN + N_VAL:
            roles[int(b)] = "val"
        else:
            roles[int(b)] = "test"
    return roles


def role_of_frame(n, seed=SPLIT_SEED):
    """Role of a 1-based frame index."""
    return block_roles(seed)[(n - 1) // BLOCK]


def frames_for(role, seed=SPLIT_SEED, n_frames=CLIP_FRAMES):
    roles = block_roles(seed, n_frames)
    return [n for n in range(1, n_frames + 1) if roles[(n - 1) // BLOCK] == role]


def mask_for(role, seed=SPLIT_SEED, n_frames=CLIP_FRAMES):
    """Boolean mask over 1..n_frames."""
    m = np.zeros(n_frames + 1, dtype=bool)
    for n in frames_for(role, seed, n_frames):
        m[n] = True
    return m


def save(seed=SPLIT_SEED):
    roles = block_roles(seed)
    payload = {
        "clip": "VeryLowLight.mp4",
        "block_frames": BLOCK,
        "n_frames": CLIP_FRAMES,
        "split_seed": seed,
        "deal": {f"n_{r}_blocks": sum(1 for b in roles if roles[b] == r)
                 for r in ("train", "val", "test")},
        "block_roles": {str(k): v for k, v in sorted(roles.items())},
        "frames": {r: len(frames_for(r, seed)) for r in ("train", "val", "test")},
        "note": ("Blocks are dealt by a seeded shuffle of block indices. All model "
                 "selection uses train + val only. The test blocks are evaluated "
                 "once and never used to choose a threshold, epoch, augmentation "
                 "setting or tolerance."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    p = save()
    print(f"seed {p['split_seed']}, block = {p['block_frames']} frames, "
          f"{p['n_frames']} frames total")
    for b, r in sorted(((int(k), v) for k, v in p["block_roles"].items())):
        lo = b * BLOCK + 1
        hi = min((b + 1) * BLOCK, CLIP_FRAMES)
        print(f"  block {b:>2}  frames {lo:>3}-{hi:<3}  {r}")
    print(f"\nframes per role: {p['frames']}")
    print(f"saved -> {OUT}")
    sys.exit(0)
