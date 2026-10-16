"""Training loop with epoch-level checkpointing of feature-reliance metrics.

Design choices (documented here because they are interview-relevant):
  * Fixed training budget, NO early stopping: condition-dependent stopping
    would confound cross-condition comparisons of training dynamics.
  * Adam with fixed LR (unless LR is the variable under test), batch 256,
    full-batch-shuffled minibatches from a dedicated seeded generator.
  * Feature-reliance metrics are evaluated at a fixed logarithmic-ish epoch
    schedule on FIXED evaluation data (test + conflict splits), so that
    preference trajectories can be compared across seeds and conditions.
    Test data is used only for measurement, never for any training or
    model-selection decision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .evaluate import conflict_preference, mlp_predict_labels
from .models import TinyMLP


def _to_tensor(x: np.ndarray, y: np.ndarray | None = None):
    xt = torch.as_tensor(x, dtype=torch.float32)
    if y is None:
        return xt
    yt = torch.as_tensor((y > 0).astype(np.float32))
    return xt, yt


def train_mlp(
    model: TinyMLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    checkpoint_epochs: list[int],
    eval_test: dict,
    eval_conflict: dict,
    weight_decay: float = 0.0,
) -> dict:
    """Train and return per-epoch loss plus checkpointed reliance metrics."""
    torch.manual_seed(seed)  # fixes dropout-free op order determinism on CPU
    gen = torch.Generator().manual_seed(seed)  # data-order generator (separate)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    x_t, y_t = _to_tensor(train_x, train_y)
    n = len(x_t)
    history = {"epoch": [], "train_loss": []}
    checkpoints: dict[int, dict] = {}

    ckpt_set = set(checkpoint_epochs)
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=gen)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / max(n_batches, 1))

        if epoch in ckpt_set:
            pred_test = mlp_predict_labels(model, eval_test["x"])
            conflicts = eval_test["u_s"] != eval_test["u_c"]
            checkpoints[epoch] = {
                "test_accuracy": float(np.mean(pred_test == eval_test["y"])),
                "conflict_preference": conflict_preference(
                    mlp_predict_labels(model, eval_conflict["x"]),
                    eval_conflict["u_s"],
                    eval_conflict["u_c"],
                )["preference_index"],
                "ablation_drop_simple_vs_complex": _quick_ablation(
                    model, eval_test
                ),
            }
    return {"history": history, "checkpoints": checkpoints}


def _quick_ablation(model: TinyMLP, test: dict, perm_seed: int = 991) -> dict:
    rng = np.random.default_rng(perm_seed)
    clean = mlp_predict_labels(model, test["x"])
    acc = float(np.mean(clean == test["y"]))
    x_s = test["x"].copy()
    x_s[:, 0] = x_s[rng.permutation(len(x_s)), 0]
    x_c = test["x"].copy()
    x_c[:, 1] = x_c[rng.permutation(len(x_c)), 1]
    acc_s = float(np.mean(mlp_predict_labels(model, x_s) == test["y"]))
    acc_c = float(np.mean(mlp_predict_labels(model, x_c) == test["y"]))
    return {"drop_simple": acc - acc_s, "drop_complex": acc - acc_c}
