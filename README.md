# Falsification Challenge — Simplicity Bias, Controlled

A small, reproducible study of the claim that **neural networks preferentially
learn simpler predictive features**. It is a *proposal* with an implemented,
executed experiment attached. The headline deliverable is
[`proposal.md`](proposal.md); this README explains how to run and reproduce
everything in it.

## Research question

> When two predictive features are equally distributed and matched in predictive
> strength but differ in functional complexity, does a small MLP preferentially
> rely on the simpler one — and if so, is that attributable to complexity itself,
> or to the complex feature being harder to learn?

## Operational definition of simplicity

Simplicity is defined in exactly one measurable way:

> **the number of decision boundaries of the feature→label function along that
> feature's input axis.**

- *Simple* feature: `u_s = sign(x_s)` → **1** boundary.
- *Complex* feature: `u_c = sign(sin(kπ·x_c))` → **2k** boundaries (`k` is a knob).

Both features are balanced, share identical input marginals (`U(−1,1)`), and are
combined into the label by a noisy threshold vote whose per-feature Bayes
accuracies are known in closed form. This is **one** definition of simplicity among
several (see `proposal.md` §9 and Appendix A); it is closest to the
low-frequency/spectral notion (Rahaman et al., 2019) and to Shah et al.'s
piecewise-linear features.

## Experimental design (one variable at a time)

Label: `y = +1 iff α·u_s + β·u_c + ε ≥ 0`, `ε ~ N(0,1)`.

| Group | Conditions | Purpose |
|---|---|---|
| **A. Baseline** | α=β=1, k=8 | matched strength; does the network prefer the simple feature? |
| **B. Strength control** | (1.3,0.7) and (0.7,1.3) at k=8 **and** k=2 | does reliance follow *strength* where the complex feature is learnable? |
| **C. Complexity sweep** | k ∈ {1,2,4,8,32}, α=β=1 | preference vs complexity; k=1 is a null anchor |
| **D. Optimisation** | isolated simple-only / complex-only at k∈{2,4,8,32}; 5× epochs; LR ×0.3/×3 | is the preference optimisation difficulty? |
| **E. Robustness** | ReLU 16×16 arm; 5 seeds everywhere | architecture / seed sensitivity |
| **F. Controls** | column-swap; shuffled labels; 4× data | positional artefact; metric sanity; data-coverage confound |

Model: MLP `2→64→64→1`, tanh, BCE-with-logits, Adam (LR 1e-3), batch 256, 100
epochs (500 for the budget control), no early stopping, 5 paired seeds per
condition.

## Purpose of each control and baseline

- **Matched strength (α=β)** — removes "the complex feature is just weaker" as an
  explanation (both Bayes accuracies equal 0.7386 by construction).
- **k=1 null anchor** — at k=1 the two features are *mathematically identical*
  (`sign(sin(πx)) ≡ sign(x)`), so a correct metric must report ~no preference.
- **Column-swap control** — proves the preference is for the simple *feature*, not
  for input column 0.
- **Isolated-learnability gradient (D1)** — measures whether the complex feature is
  learnable *at all* in isolation; this is the crux of the "ease of optimisation"
  alternative explanation.
- **Budget / LR controls** — rule out "it just needs longer" / step-size artefacts.
- **Shuffled-label control** — sanity check: no signal ⇒ ~chance accuracy and no
  *consistent* preference.
- **Logistic-regression baseline** — shows what simplicity looks like when *imposed
  by model class* (a linear model can only use the linearly-separable feature).
- **Analytic oracle** — the exact Bayes accuracies and true per-feature
  contribution, computed from closed forms (`src/generate_data.py`), verified
  against Monte-Carlo in the test suite.

## Metrics (fixed before running)

- **Conflict preference index P (primary).** On examples where the features
  disagree (`u_s ≠ u_c`), `P = frac(pred = u_s) − frac(pred = u_c) ∈ [−1,1]`.
  `P>0` = follows simple, `P<0` = follows complex. Label-free.
- **Permutation-ablation drops** — `drop_s`, `drop_c`. Clean here because the two
  input columns are independent *by construction*, so permuting one does not move
  the input distribution.
- **Isolated test accuracy** vs the analytic Bayes value.

## Installation

Requires Python 3.10+ (developed on 3.12.3). From this directory:

```bash
python -m pip install -r requirements.txt
```

Dependencies: NumPy, SciPy, scikit-learn, PyTorch (CPU build is enough),
Matplotlib, pytest.

## How to run

All commands are run **from this directory** (`falsification-challenge/`).

```bash
# 1) unit tests (data generator + metrics)
python -m pytest -q

# 2) full experiment suite (all conditions, 5 seeds each; ~20-30 min on CPU)
python -m src.run_experiments

# 3) run only specific conditions (results on disk are always included in summary)
python -m src.run_experiments --conditions baseline,k2,k4

# 4) aggregate results into results/analysis_summary.md
python -m src.analyze_results

# 5) regenerate figures from recorded results
python -m src.plot_results
```

Training is deterministic given the seeds; re-running produces identical metrics
on the same platform. `results/run_manifest.json` records library versions and
timings.

## Where results are saved

```
results/
  runs/<condition>_seed<s>.json   per-run settings, metrics, learning curves
  summary.json                    per-condition aggregates (mean ± SD over seeds)
  reference_metrics.json          analytic oracles per (α, β, k)
  analysis_summary.md             generated human-readable table
  run_manifest.json               config, library versions, timings
  figures/*.png                   five figures (each answers one question)
```

## Implemented vs planned

**Implemented and executed:** data generator with analytic oracle; MLP and logistic
baselines; training with checkpointed reliance tracking; the full condition grid;
unit tests (data + metrics); aggregation; figures. See `proposal.md` §8 for the
**actual** results and `results/analysis_summary.md` for the generated table.

**Not implemented (noted as future work):**
- Forcing the network onto the complex feature (e.g. masking `x_s`) to test whether
  a *learnable* complex feature is then used.
- Warm-up / feature-widening schedules to test whether the easy feature's head
  start is the whole story.
- Additional operationalisations of simplicity (linear separability, low
  dimensionality per Morwani et al., description length).
- Correlated (rather than independent) features, and tasks with genuine feature
  *synergy*.
- Larger model families (CNN/transformer) and full hyperparameter sweeps.

## Known limitations

- One definition of simplicity; the complexity knob confounds frequency, region
  count and samples-per-region (bounded, not removed, by the 4×-data control).
- The feature vote is non-combinable: the task is "pick a feature", not "combine".
- Small scale: one MLP family, 5 seeds, one LR grid — the learnability boundary is
  a property of (feature, architecture, optimiser, budget), not of the feature alone.
- Non-trivial seed variance under the null (shuffled labels, k=1) — we lean on
  effect sizes that exceed this spread, and we do not over-interpret small
  per-seed differences.

## Suggested next experiments

1. Mask the simple feature at test/train time and measure whether a *learnable*
   complex feature is used when it is the only option.
2. Add a warm-up phase that delays the simple feature, then compare final reliance
   — isolating "head start" from "intrinsic preference".
3. Repeat the k-sweep under a second simplicity definition (linear separability)
   and check whether the learnability boundary moves in lock-step.
4. Scale the model (width/depth, then a small CNN) to see whether the learnability
   boundary at k=8 shifts.

## Status honesty

- The experiment **was run**; the numbers in `proposal.md` §8 come from
  `results/`. Section 7 of the proposal is the pre-registered set of predictions
  and is kept separate from the observed results.
- The test suite passes (21 tests). Counts and figures are those actually present
  on disk; nothing here is fabricated.
