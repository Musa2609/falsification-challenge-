"""Tests for synthetic data generation: correctness of the controlled design.

These tests verify the properties the whole experiment depends on:
  1. Both features are balanced and share identical input marginals.
  2. At k = 1 the complex feature is *identical* to the simple one (null anchor).
  3. Labels are generated exactly by the documented formula.
  4. Generation is deterministic and paired conditions share their inputs.
  5. Splits are disjoint (no leakage).
  6. Analytic Bayes accuracies match Monte-Carlo estimates.
  7. The shuffled-label control destroys the feature-label relationship.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.generate_data import (
    bayes_joint_accuracy,
    bayes_single_accuracy,
    complex_feature,
    generate_dataset,
    shuffled_labels,
    simple_feature,
)


def make(n_train=2000, n_val=500, n_test=2000, n_conflict=1000, **kw):
    defaults = dict(base_seed=7, alpha=1.0, beta=1.0, k=8)
    defaults.update(kw)
    return generate_dataset(
        **defaults, n_train=n_train, n_val=n_val, n_test=n_test,
        n_conflict_test=n_conflict,
    )


def test_features_are_balanced_and_share_marginals():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(100_000, 2))
    u_s = simple_feature(x[:, 0])
    for k in [1, 2, 8, 32]:
        u_c = complex_feature(x[:, 1], k)
        assert abs(np.mean(u_s) - 0.0) < 0.01
        assert abs(np.mean(u_c) - 0.0) < 0.01
        assert set(np.unique(u_c)) == {-1.0, 1.0}


def test_k1_complex_feature_equals_simple_threshold():
    # u_c = sign(sin(pi*x)) == sign(x) on (-1, 1): exact null-calibration anchor.
    x = np.linspace(-0.999, 0.999, 9999)
    np.testing.assert_array_equal(complex_feature(x, 1), simple_feature(x))


def test_label_formula_is_exact():
    # Recompute the label from the stored features and the recorded score; the
    # generator uses the same RNG stream, so we re-derive using an independent
    # check: sign agreement when noise is absent is deterministic.
    d = make(base_seed=11, alpha=1.3, beta=0.7, noise_sigma=0.0)
    score = 1.3 * d["train"]["u_s"] + 0.7 * d["train"]["u_c"]
    expected = np.where(score >= 0, 1.0, -1.0)
    np.testing.assert_array_equal(d["train"]["y"], expected)


def test_generation_is_deterministic():
    a = make(base_seed=3)
    b = make(base_seed=3)
    for split in ["train", "val", "test", "conflict_test"]:
        np.testing.assert_array_equal(a[split]["x"], b[split]["x"])
        np.testing.assert_array_equal(a[split]["y"], b[split]["y"])


def test_paired_conditions_share_inputs_and_noise():
    # Same base_seed, different k -> identical inputs and (because the RNG
    # stream order is fixed) identical shared-label-noise draws for train/val/
    # test. Only the feature DEFINITION changes, which is what makes seed-paired
    # comparisons across conditions valid.
    a = make(base_seed=5, k=2)
    b = make(base_seed=5, k=32)
    for split in ["train", "val", "test"]:
        np.testing.assert_array_equal(a[split]["x"], b[split]["x"])
        # u_s does not depend on k, so it must be identical too.
        np.testing.assert_array_equal(a[split]["u_s"], b[split]["u_s"])
    # NOTE: the conflict-only evaluation set is rejection-sampled using u_c,
    # whose definition depends on k, so its rows are NOT shared across k. This
    # is expected and harmless: it is evaluation-only and independent of the
    # training data.


def test_splits_are_disjoint_no_leakage():
    d = make(base_seed=9)
    keys = {name: set(map(tuple, np.round(d[name]["x"], 10))) for name in ["train", "test", "conflict_test"]}
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["train"].isdisjoint(keys["conflict_test"])
    assert keys["test"].isdisjoint(keys["conflict_test"])


def test_conflict_set_contains_only_conflicts():
    d = make(base_seed=2)
    assert np.all(d["conflict_test"]["u_s"] != d["conflict_test"]["u_c"])


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (1.3, 0.7), (0.7, 1.3), (1.5, 0.5)])
def test_analytic_bayes_matches_monte_carlo(alpha, beta):
    d = generate_dataset(
        base_seed=123, alpha=alpha, beta=beta, k=8,
        n_train=0, n_val=0, n_test=400_000, n_conflict_test=4,
    )
    t = d["test"]
    mc_simple = np.mean(simple_feature(t["x"][:, 0]) == t["y"])
    mc_complex = np.mean(complex_feature(t["x"][:, 1], 8) == t["y"])
    assert abs(mc_simple - bayes_single_accuracy(alpha, beta, "simple")) < 0.005
    assert abs(mc_complex - bayes_single_accuracy(alpha, beta, "complex")) < 0.005


def test_matched_strength_iff_alpha_equals_beta():
    assert abs(bayes_single_accuracy(1.0, 1.0, "simple")
               - bayes_single_accuracy(1.0, 1.0, "complex")) < 1e-12
    assert bayes_single_accuracy(1.3, 0.7, "simple") > bayes_single_accuracy(1.3, 0.7, "complex")
    assert bayes_single_accuracy(0.7, 1.3, "complex") > bayes_single_accuracy(0.7, 1.3, "simple")


def test_joint_equals_better_single_feature():
    # In this vote model features are non-combinable; document it as a property.
    assert abs(bayes_joint_accuracy(1.3, 0.7)
               - bayes_single_accuracy(1.3, 0.7, "simple")) < 1e-12


def test_shuffled_labels_break_signal():
    d = make(base_seed=4)
    y = d["train"]["y"]
    y_shuf = shuffled_labels(y, 4)
    assert np.array_equal(np.sort(y), np.sort(y_shuf))  # same multiset
    assert np.mean(y_shuf == y) < 0.55            # essentially decorrelated
    # The shuffled labels must be uncorrelated with BOTH features: this is what
    # makes the shuffled-label condition a valid no-signal control.
    corr_s = np.corrcoef(y_shuf, d["train"]["u_s"])[0, 1]
    corr_c = np.corrcoef(y_shuf, d["train"]["u_c"])[0, 1]
    assert abs(corr_s) < 0.05
    assert abs(corr_c) < 0.05
