"""Model definitions: small MLP (PyTorch) and a logistic-regression baseline.

All models take the raw 2-D input (x_s, x_c) and output a single logit.
The MLP is deliberately tiny (two hidden layers) and conventional; nothing in
the study depends on its capacity, only on its training procedure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


class TinyMLP(nn.Module):
    def __init__(self, width=(64, 64), activation: str = "tanh"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        layers: list[nn.Module] = []
        in_dim = 2
        for w in width:
            layers += [nn.Linear(in_dim, w), ACTIVATIONS[activation]()]
            in_dim = w
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_mlp(width=(64, 64), activation: str = "tanh", seed: int = 0) -> TinyMLP:
    """Build an MLP with a deterministic initialization.

    torch.manual_seed fixes the initialization; initialization is the ONLY
    thing this seed controls (data order in training has its own generator).
    """
    torch.manual_seed(seed)
    model = TinyMLP(width=tuple(width), activation=activation)
    model.init_weights()
    return model


def _init_weights(self: nn.Module) -> None:
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


TinyMLP.init_weights = _init_weights  # type: ignore[attr-defined]


def fit_logistic_baseline(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Linear baseline: what simplicity-by-model-class-restriction looks like.

    Near-unregularized (large C) so the fit is essentially unpenalized MLE.
    Deterministic solver (lbfgs) -> no seed dependence.
    """
    clf = LogisticRegression(max_iter=10_000, C=1e6, solver="lbfgs")
    clf.fit(x, np.where(y > 0, 1, 0))
    return clf


def logistic_predict(clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    """Predict labels in {-1, +1}."""
    return np.where(clf.predict(x) == 1, 1.0, -1.0)
