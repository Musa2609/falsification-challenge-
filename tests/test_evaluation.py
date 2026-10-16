"""Tests for the feature-reliance metrics.

The metrics are validated against predictors with KNOWN reliance:
  * an oracle that uses only the simple feature must show preference +1 and a
    near-zero accuracy drop when the complex column is permuted;
  * an oracle that uses only the complex feature must show preference -1;
  * a constant predictor must show preference ~0 and zero drops;
  * permutation ablation must not change the marginal distribution of a column.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    conflict_preference,
    evaluate_reference_predictor,
    permutation_ablation_drops,
)
from src.generate_data import complex_feature, generate_dataset, simple_feature


@pytest.fixture()
def data():
    return generate_dataset(
        base_seed=21, alpha=1.0, beta=1.0, k=8,
        n_train=0, n_val=0, n_test=2000, n_conflict_test=2000,
    )


def test_oracle_simple_prefers_simple_feature(data):
    pred = simple_feature(data["conflict_test"]["x"][:, 0])
    out = conflict_preference(pred, data["conflict_test"]["u_s"], data["conflict_test"]["u_c"])
    assert out["preference_index"] == pytest.approx(1.0)


def test_oracle_complex_prefers_complex_feature(data):
    pred = complex_feature(data["conflict_test"]["x"][:, 1], 8)
    out = conflict_preference(pred, data["conflict_test"]["u_s"], data["conflict_test"]["u_c"])
    assert out["preference_index"] == pytest.approx(-1.0)


def test_constant_predictor_has_no_preference(data):
    pred = np.ones(len(data["conflict_test"]["x"]))
    out = conflict_preference(pred, data["conflict_test"]["u_s"], data["conflict_test"]["u_c"])
    assert abs(out["preference_index"]) < 0.05


def test_conflict_preference_rejects_non_conflict_input():
    u = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        conflict_preference(np.array([1.0, -1.0]), u, u)


def test_ablation_drops_identify_used_feature(data):
    t = data["test"]
    rng = np.random.default_rng(0)
    clean = simple_feature(t["x"][:, 0])
    x = t["x"].copy()
    x[:, 0] = x[rng.permutation(len(x)), 0]
    pred_no_s = simple_feature(x[:, 0])
    x2 = t["x"].copy()
    x2[:, 1] = x2[rng.permutation(len(x2)), 1]
    pred_no_c = simple_feature(x2[:, 0])
    out = permutation_ablation_drops(clean, pred_no_s, pred_no_c, t["y"])
    assert out["drop_simple"] > 0.1          # destroying the used feature hurts
    assert out["drop_complex"] == pytest.approx(0.0, abs=1e-9)  # unused feature: no effect


def test_oracle_simple_evaluation_bundle(data):
    out = evaluate_reference_predictor(
        lambda z: simple_feature(z[:, 0]), data["test"], data["conflict_test"], perm_seed=1
    )
    # Accuracy must equal the analytic single-feature Bayes accuracy (alpha=beta=1).
    assert out["test_accuracy"] == pytest.approx(0.5 * 0.9766 + 0.25, abs=0.02)
    assert out["conflict_preference"]["preference_index"] == pytest.approx(1.0)


def test_permutation_preserves_column_marginal():
    x = np.random.default_rng(0).uniform(-1, 1, size=(5000, 2))
    perm = np.random.default_rng(1).permutation(len(x))
    x2 = x.copy()
    x2[:, 0] = x2[perm, 0]
    # Sort-based comparison of the marginal distribution.
    np.testing.assert_allclose(np.sort(x[:, 0]), np.sort(x2[:, 0]))
    # Independence is a design property of the data generator, not of the permutation.
    assert abs(np.corrcoef(x2[:, 0], x2[:, 1])[0, 1]) < 0.05
