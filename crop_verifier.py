"""Hand-vs-clutter crop verifier: model definition + inference factory.

This is the learned replacement for the box-size FP gate.  It answers "is this
detector crop actually a hand?" from the crop alone, so it works at any range --
which the size gate structurally cannot.

Consumed by hand_mouse_cursor.py (as the in-window gate) and by
train_cnn_verifier.py (training).  Kept in its own module so a frozen build
picks it up through a normal static import.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

SIZE = 64           # network input (crops are stored at 128)


class SmallCNN(nn.Module):
    """~50k-parameter CNN.  Deliberately small: the task is binary and the
    input is a small, already rotation-normalised crop."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.body(x).flatten(1)).squeeze(-1)


def preprocess(crop_bgr):
    """BGR crop (any size) -> normalised NCHW tensor."""
    img = cv2.resize(crop_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def load_model(path):
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def make_verifier(model_path):
    """Return a callable(crop_bgr) -> P(hand), per GATE_REPLACEMENT_DESIGN.md."""
    model = load_model(model_path)

    def verify(crop_bgr):
        with torch.no_grad():
            return float(torch.sigmoid(model(preprocess(crop_bgr))))
    return verify
