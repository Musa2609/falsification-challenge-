"""Feature-reliance evaluation metrics.

Primary metric: conflict preference index P in [-1, 1], measured on test
examples where the two features disagree (u_s != u_c):
    P = frac(predictions == u_s) - frac(predictions == u_c)
P > 0: the model sides with the simple feature on conflicts.
P < 0: the model sides with the complex feature.
P ~ 0: no systematic preference (or reliance on both / neither).
This metric is label-free: it asks which FEATURE the model follows, not which
is correct (on conflicts at matched strength the true label is pure noise).

Secondary metric: permutation-ablation reliance.
    drop_s = acc(X) - acc(X with column 0 permuted)
    drop_c = acc(X) with column 1 permuted)
Because the two input columns are independent BY CONSTRUCTION, permuting one
column preserves the input joint distribution and only destroys that
feature's information — a rare property this design deliberately exploits.
"""
from __future__ import annotations

import numpy as np
import torch

from .models import TinyMLP


def mlp_predict_labels(model: TinyMLP, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Predict {-1, +1} labels for an (n, 2) array with a trained MLP."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch], dtype=torch.float32)
            logits = model(xb)
            out.append((torch.sigmoid(logits) >= 0.5).numpy())
    return np.where(np.concatenate(out), 1.0, -1.0)


def conflict_preference(pred: np.ndarray, u_s: np.ndarray, u_c: np.ndarray) -> dict:
    """Preference index on conflict examples (u_s != u_c by construction)."""
    if not np.all(u_s != u_c):
        raise ValueError("conflict set must contain only u_s != u_c examples")
    frac_simple = float(np.mean(pred == u_s))
    frac_complex = float(np.mean(pred == u_c))
    return {
        "frac_follows_simple": frac_simple,
        "frac_follows_complex": frac_complex,
        "preference_index": frac_simple - frac_complex,  # P in [-1, 1]
        "n": int(len(pred)),
    }


def permutation_ablation_drops(
    pred_clean: np.ndarray,
    pred_ablated_s: np.ndarray,
    pred_ablated_c: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Accuracy drops when each feature column is independently permuted."""
    acc_clean = float(np.mean(pred_clean == y))
    acc_no_s = float(np.mean(pred_ablated_s == y))
    acc_no_c = float(np.mean(pred_ablated_c == y))
    return {
        "acc_clean": acc_clean,
        "acc_no_simple": acc_no_s,
        "acc_no_complex": acc_no_c,
        "drop_simple": acc_clean - acc_no_s,
        "drop_complex": acc_clean - acc_no_c,
        "relative_reliance": (acc_clean - acc_no_c) - (acc_clean - acc_no_s),  # >0: leans simple
    }


def ablated_predictions(
    model_predict_fn, x: np.ndarray, perm_rng: np.random.Generator
) -> dict:
    """Produce clean and ablated predictions with one shared permutation per column."""
    pred_clean = model_predict_fn(x)
    perm_s = perm_rng.permutation(len(x))
    perm_c = perm_rng.permutation(len(x))
    x_no_s = x.copy()
    x_no_s[:, 0] = x_no_s[perm_s, 0]
    x_no_c = x.copy()
    x_no_c[:, 1] = x_no_c[perm_c, 1]
    return {
        "pred_clean": pred_clean,
        "pred_ablated_s": model_predict_fn(x_no_s),
        "pred_ablated_c": model_predict_fn(x_no_c),
    }


def evaluate_mlp(model: TinyMLP, test: dict, conflict: dict, perm_seed: int) -> dict:
    """Full evaluation bundle for one trained MLP on one dataset."""
    fn = lambda z: mlp_predict_labels(model, z)
    preds = ablated_predictions(fn, test["x"], np.random.default_rng(perm_seed))
    abl = permutation_ablation_drops(
        preds["pred_clean"], preds["pred_ablated_s"], preds["pred_ablated_c"], test["y"]
    )
    pref = conflict_preference(fn(conflict["x"]), conflict["u_s"], conflict["u_c"])
    pref_test_conflicts = conflict_preference(
        fn(test["x"][test["u_s"] != test["u_c"]]),
        test["u_s"][test["u_s"] != test["u_c"]],
        test["u_c"][test["u_s"] != test["u_c"]],
    )
    return {
        "test_accuracy": abl["acc_clean"],
        "ablation": abl,
        "conflict_preference": pref,
        "conflict_preference_on_test_split": pref_test_conflicts,
    }


def evaluate_reference_predictor(
    predict_fn, test: dict, conflict: dict, perm_seed: int
) -> dict:
    """Same evaluation bundle for any {-1,+1} predictor (oracles, logreg)."""
    preds = ablated_predictions(predict_fn, test["x"], np.random.default_rng(perm_seed))
    abl = permutation_ablation_drops(
        preds["pred_clean"], preds["pred_ablated_s"], preds["pred_ablated_c"], test["y"]
    )
    pref = conflict_preference(predict_fn(conflict["x"]), conflict["u_s"], conflict["u_c"])
    return {"test_accuracy": abl["acc_clean"], "ablation": abl, "conflict_preference": pref}
