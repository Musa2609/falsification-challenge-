"""Figures from recorded results only. Reads results/summary.json and the
per-run JSON files; never fabricates measurements.

Figures produced (each answers one predefined question):
  1. fig_preference_vs_k.png        -- does preference follow feature complexity?
  2. fig_strength_control.png       -- does preference follow predictive strength?
  3. fig_learning_curves.png        -- does preference change with training time?
  4. fig_single_feature_learnability.png -- is the complex feature simply slower
                                            to learn in isolation?

Usage: python -m src.plot_results
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"


def load_summary():
    return json.loads((RESULTS / "summary.json").read_text())


def load_runs():
    runs = []
    for p in sorted((RESULTS / "runs").glob("*.json")):
        runs.append(json.loads(p.read_text()))
    return runs


def mean_std(vals):
    a = np.array(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def fig_preference_vs_k(runs):
    order = ["k1_null", "k2", "k4", "baseline", "k32"]
    labels = ["k=1\n(null anchor)", "k=2", "k=4", "k=8\n(baseline)", "k=32"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, cond in enumerate(order):
        vals, _ = zip(*[(r["final_eval"]["conflict_preference"]["preference_index"], 0)
                        for r in runs if r["condition"] == cond])
        ax.scatter([i] * len(vals), vals, s=32, alpha=0.7, zorder=3)
        m, s = mean_std(vals)
        ax.errorbar(i, m, yerr=s, fmt="o", color="black", capsize=4, zorder=4)
    ax.axhline(0, color="grey", lw=1)
    ax.axhline(1, color="red", ls=":", lw=1)
    ax.axhline(-1, color="blue", ls=":", lw=1)
    ax.text(4.35, 0.95, "always\nsimple", fontsize=8, color="red")
    ax.text(4.35, -1.05, "always\ncomplex", fontsize=8, color="blue")
    ax.set_xlim(-0.5, 5.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("conflict preference P\n(+1 = follows simple, -1 = follows complex)")
    ax.set_title("Preference vs feature complexity (alpha = beta, matched strength)")
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_preference_vs_k.png", dpi=150)
    plt.close(fig)


def fig_strength_control(runs):
    """Does preference follow predictive STRENGTH, and does that depend on
    whether the complex feature is learnable (k=2) or not (k=8)?"""
    groups = [
        ("simple stronger\n(1.3, 0.7)", "strength_simple_favored"),
        ("matched\n(1.0, 1.0)", "baseline"),
        ("complex stronger\n(0.7, 1.3)", "strength_complex_favored"),
    ]
    groups_k2 = [
        ("simple stronger\n(1.3, 0.7)", "strength_simple_favored_k2"),
        ("matched\n(1.0, 1.0)", "k2"),
        ("complex stronger\n(0.7, 1.3)", "strength_complex_favored_k2"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for offset, grp, color, lab in [
        (-0.16, groups, "tab:blue", "k = 8 (complex unlearnable)"),
        (0.16, groups_k2, "tab:green", "k = 2 (complex learnable)"),
    ]:
        for i, (_xlabel, cond) in enumerate(grp):
            vals = [r["final_eval"]["conflict_preference"]["preference_index"]
                    for r in runs if r["condition"] == cond]
            ax.scatter([i + offset] * len(vals), vals, s=26, alpha=0.6, color=color, zorder=3)
            m, s = mean_std(vals)
            ax.errorbar(i + offset, m, yerr=s, fmt="o", color="black", capsize=4, zorder=4)
        ax.scatter([], [], color=color, label=lab)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlabel("relative predictive strength (alpha, beta)")
    ax.set_ylabel("conflict preference P")
    ax.set_title("Preference vs predictive strength, at k=8 vs k=2")
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_strength_control.png", dpi=150)
    plt.close(fig)


def fig_positional_control(runs):
    """Direct positional control: swap which input column holds the simple vs
    complex feature. Preference must stay tied to the simple FEATURE."""
    order = ["baseline", "baseline_swapped"]
    labels = ["baseline\n(simple = column 0)", "swapped\n(simple = column 1)"]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for i, cond in enumerate(order):
        vals = [r["final_eval"]["conflict_preference"]["preference_index"]
                for r in runs if r["condition"] == cond]
        ax.scatter([i] * len(vals), vals, s=30, alpha=0.7, zorder=3)
        m, s = mean_std(vals)
        ax.errorbar(i, m, yerr=s, fmt="o", color="black", capsize=4, zorder=4)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("conflict preference P")
    ax.set_title("Positional control (k = 8, matched strength)")
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_positional_control.png", dpi=150)
    plt.close(fig)


def fig_learning_curves(runs):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for cond, color in [("baseline", "tab:blue"), ("longrun_5x", "tab:orange")]:
        runs_c = [r for r in runs if r["condition"] == cond]
        by_epoch: dict[int, list[float]] = {}
        for r in runs_c:
            for e, ck in r["checkpoints"].items():
                by_epoch.setdefault(int(e), []).append(ck["conflict_preference"])
        xs = sorted(by_epoch)
        ms = [np.mean(by_epoch[e]) for e in xs]
        lo = [np.min(by_epoch[e]) for e in xs]
        hi = [np.max(by_epoch[e]) for e in xs]
        ax.plot(xs, ms, "-o", ms=3, label=f"{cond} (mean)", color=color)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color, label=f"{cond} (range)")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("conflict preference P")
    ax.set_title("Preference during training (baseline vs 5x budget)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_learning_curves.png", dpi=150)
    plt.close(fig)


def fig_single_feature_learnability(runs):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for cond, color in [("single_simple", "tab:green"), ("single_complex", "tab:red")]:
        runs_c = [r for r in runs if r["condition"] == cond]
        by_epoch: dict[int, list[float]] = {}
        for r in runs_c:
            for e, ck in r["checkpoints"].items():
                by_epoch.setdefault(int(e), []).append(ck["test_accuracy"])
        xs = sorted(by_epoch)
        ms = [np.mean(by_epoch[e]) for e in xs]
        lo = [np.min(by_epoch[e]) for e in xs]
        hi = [np.max(by_epoch[e]) for e in xs]
        ax.plot(xs, ms, "-o", ms=3, label=f"{cond} (mean)", color=color)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("Learnability in isolation: simple-only vs complex-only labels")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_single_feature_learnability.png", dpi=150)
    plt.close(fig)


def fig_preference_and_learnability_vs_k(runs):
    """The central figure: joint-task preference vs k (top) and the complex
    feature's ISOLATED learnability vs k (bottom). If preference tracks
    learnability, the top curve should fall exactly where the bottom curve
    is low. Bayes accuracy for the isolation task (label depends on u_c with
    strength 1) is Phi(1) = 0.8413."""
    from scipy.stats import norm

    ks = [1, 2, 4, 8, 32]
    joint_names = {1: "k1_null", 2: "k2", 4: "k4", 8: "baseline", 32: "k32"}
    single_names = {2: "single_complex_k2", 4: "single_complex_k4",
                    8: "single_complex", 32: "single_complex_k32"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True)

    for k in ks:
        vals = [r["final_eval"]["conflict_preference"]["preference_index"]
                for r in runs if r["condition"] == joint_names[k]]
        ax1.scatter([k] * len(vals), vals, s=28, alpha=0.65, zorder=3)
        m, s = mean_std(vals)
        ax1.errorbar(k, m, yerr=s, fmt="o", color="black", capsize=4, zorder=4)
    ax1.axhline(0, color="grey", lw=1)
    ax1.set_ylabel("joint-task preference P\n(+1 simple, -1 complex)")
    ax1.set_title("Preference vs complexity, and the complex feature's isolated learnability")
    ax1.set_ylim(-1.15, 1.15)

    for k in [2, 4, 8, 32]:
        vals = [r["final_eval"]["test_accuracy"]
                for r in runs if r["condition"] == single_names[k]]
        ax2.scatter([k] * len(vals), vals, s=28, alpha=0.65, zorder=3)
        m, s = mean_std(vals)
        ax2.errorbar(k, m, yerr=s, fmt="o", color="tab:red", capsize=4, zorder=4)
    ax2.axhline(0.5, color="grey", lw=1, ls="--")
    ax2.axhline(float(norm.cdf(1.0)), color="green", lw=1, ls=":")
    ax2.text(20, float(norm.cdf(1.0)) + 0.01, "Bayes (isolated complex)", fontsize=8, color="green")
    ax2.text(20, 0.51, "chance", fontsize=8, color="grey")
    ax2.set_xlabel("feature complexity k (number of decision boundaries = 2k)")
    ax2.set_ylabel("isolated complex-only\ntest accuracy")
    ax2.set_ylim(0.45, 0.9)
    ax2.set_xscale("log")
    ax2.set_xticks([1, 2, 4, 8, 32])
    ax2.set_xticklabels(["1", "2", "4", "8", "32"])

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_preference_and_learnability_vs_k.png", dpi=150)
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("No runs found in results/runs/ — run the experiment first.")
    fig_preference_vs_k(runs)
    fig_strength_control(runs)
    fig_positional_control(runs)
    fig_learning_curves(runs)
    fig_single_feature_learnability(runs)
    fig_preference_and_learnability_vs_k(runs)
    print(f"Figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
