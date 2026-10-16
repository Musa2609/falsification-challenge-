"""Synthetic two-feature dataset with analytically controlled properties.

Design (see proposal.md):
  - Inputs: x = (x_s, x_c), two iid Uniform(-1, 1) columns.
  - Simple feature:  u_s = sign(x_s)                    -> 1 decision boundary.
  - Complex feature: u_c = sign(sin(k * pi * x_c))      -> 2k decision boundaries.
  - Label:           y = +1 iff alpha*u_s + beta*u_c + N(0, sigma^2) >= 0.

Properties that make this design controllable:
  * Both u_s and u_c are balanced Bernoulli(1/2) over the input domain for any
    integer k >= 1, with identical input marginals. Changing k changes ONLY the
    functional complexity of the x_c -> label map.
  * Single-feature Bayes accuracies are exact closed forms:
        A_s = 0.5*Phi(alpha+beta) + 0.5*Phi(alpha-beta)
        A_c = 0.5*Phi(alpha+beta) + 0.5*Phi(beta-alpha)
    so predictive strength is measurable, and A_s == A_c iff alpha == beta.
  * At k = 1, u_c == sign(x_c) identically: the two features become
    mathematically interchangeable (null-calibration anchor for preference
    metrics).
  * At alpha == beta, on conflict examples (u_s != u_c) the label is pure
    noise, so neither feature helps there; the two features are equally useful
    overall and there is no statistical pressure toward either one. Any
    systematic side-taking by the model reflects inductive/optimization bias.

RNG policy: each run gets a base SeedSequence; independent child streams are
spawned for (inputs, label noise, conflict-set, evaluation permutation) so
that paired conditions (same seed, different k/alpha/beta) reuse the exact same
input and noise draws. See generate_dataset().
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def simple_feature(x: np.ndarray) -> np.ndarray:
    """u_s = sign(x_s). One decision boundary at 0."""
    return np.where(x >= 0.0, 1.0, -1.0)


def complex_feature(x: np.ndarray, k: int) -> np.ndarray:
    """u_c = sign(sin(k*pi*x)) on x in (-1, 1). 2k decision boundaries."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return np.where(np.sin(k * np.pi * x) >= 0.0, 1.0, -1.0)


def bayes_single_accuracy(alpha: float, beta: float, feature: str) -> float:
    """Exact Bayes accuracy of the best classifier using ONE feature only.

    feature='simple' -> uses u_s; feature='complex' -> uses u_c.
    A_s = 0.5*Phi(alpha+beta) + 0.5*Phi(alpha-beta); A_c swaps alpha/beta.
    """
    if feature == "simple":
        return float(0.5 * norm.cdf(alpha + beta) + 0.5 * norm.cdf(alpha - beta))
    if feature == "complex":
        return float(0.5 * norm.cdf(alpha + beta) + 0.5 * norm.cdf(beta - alpha))
    raise ValueError("feature must be 'simple' or 'complex'")


def bayes_joint_accuracy(alpha: float, beta: float) -> float:
    """Exact Bayes accuracy of the optimal classifier using BOTH features.

    In this vote model the features are non-combinable: joint accuracy equals
    the better single-feature accuracy (on conflicts the signal imbalance
    |alpha-beta| is all that remains).
    """
    return float(0.5 * norm.cdf(alpha + beta) + 0.5 * norm.cdf(abs(alpha - beta)))


def _sample_features_and_inputs(rng: np.random.Generator, n: int, k: int):
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    u_s = simple_feature(x[:, 0])
    u_c = complex_feature(x[:, 1], k)
    return x, u_s, u_c


def _sample_labels(
    rng: np.random.Generator,
    u_s: np.ndarray,
    u_c: np.ndarray,
    alpha: float,
    beta: float,
    label_mode: str,
    noise_sigma: float,
) -> np.ndarray:
    if label_mode == "both":
        score = alpha * u_s + beta * u_c
    elif label_mode == "simple":
        score = alpha * u_s
    elif label_mode == "complex":
        score = beta * u_c
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")
    noise = rng.normal(0.0, noise_sigma, size=u_s.shape[0])
    return np.where(score + noise >= 0.0, 1.0, -1.0)


def generate_dataset(
    base_seed: int,
    alpha: float,
    beta: float,
    k: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_conflict_test: int,
    label_mode: str = "both",
    noise_sigma: float = 1.0,
) -> dict:
    """Generate all splits for one (seed, condition) run.

    Independent child RNG streams are spawned from base_seed so that splits do
    not share state. Paired conditions that share base_seed receive identical
    input and noise draws (only the feature definition / coefficients differ).
    """
    ss = np.random.SeedSequence(base_seed)
    rng_inputs, rng_noise, rng_conflict = [np.random.default_rng(s) for s in ss.spawn(3)]

    splits = {}
    for name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        x, u_s, u_c = _sample_features_and_inputs(rng_inputs, n, k)
        y = _sample_labels(rng_noise, u_s, u_c, alpha, beta, label_mode, noise_sigma)
        splits[name] = {"x": x, "u_s": u_s, "u_c": u_c, "y": y}

    # Conflict-only evaluation set: rejection-sample examples where u_s != u_c.
    # Evaluation-only; never seen by training or model selection.
    accepted_x, accepted_us, accepted_uc = [], [], []
    remaining = n_conflict_test
    while remaining > 0:
        x, u_s, u_c = _sample_features_and_inputs(rng_conflict, max(remaining * 2, 1024), k)
        mask = u_s != u_c
        accepted_x.append(x[mask])
        accepted_us.append(u_s[mask])
        accepted_uc.append(u_c[mask])
        remaining -= int(mask.sum())
    x_conf = np.concatenate(accepted_x)[:n_conflict_test]
    us_conf = np.concatenate(accepted_us)[:n_conflict_test]
    uc_conf = np.concatenate(accepted_uc)[:n_conflict_test]
    # Labels on the conflict set are noise at alpha==beta and are NOT used by
    # the preference metric (which is label-free); kept for completeness.
    y_conf = _sample_labels(rng_noise, us_conf, uc_conf, alpha, beta, label_mode, noise_sigma)
    splits["conflict_test"] = {
        "x": x_conf, "u_s": us_conf, "u_c": uc_conf, "y": y_conf,
    }
    return splits


def shuffled_labels(y: np.ndarray, base_seed: int) -> np.ndarray:
    """Permutation of labels: destroys all feature-label signal (sanity control)."""
    rng = np.random.default_rng(np.random.SeedSequence(base_seed).spawn(1)[0])
    out = y.copy()
    rng.shuffle(out)
    return out
