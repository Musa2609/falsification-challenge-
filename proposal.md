# Simplicity or Learnability? A Controlled Test of Whether Neural Networks Preferentially Learn Simpler Predictive Features

*Falsification Challenge — research proposal with a small implemented experiment.*

> **How to read this document.** Following the challenge brief, the proposal is
> written first (Sections 1–7). Section 8 reports *actual* experimental results
> from a small implementation that was run; it is kept strictly separate from the
> pre-registered expectations in Section 7. Section 11 lists decisions that are
> *yours to own* before submission — the reasoning below is meant to be reviewed
> and defended, not accepted blindly.

---

## Abstract

The claim under test is that "neural networks preferentially learn simpler
predictive features over more complex ones." The hard part is that *simple* is
not a single concept. We operationalise simplicity in exactly one measurable way
— the **number of decision boundaries of the feature→label function along that
feature's input axis** — and then ask whether an observed preference for the
simple feature survives when its **predictive strength** is controlled and when
the **optimisation difficulty** of the complex feature is measured directly. We
build a synthetic binary task with two balanced, equally-distributed features: a
simple one (one boundary) and a complex one (2k boundaries, k a knob), combined
by a noisy threshold vote whose per-feature Bayes accuracies are known in closed
form. Using a tiny MLP (2→64→64→1, Adam), 5 paired seeds per condition, and a
label-free "which feature does the network follow?" metric on examples where the
two features disagree, the implemented experiment finds that the preference is
**real but conditional**: it is near-total when the complex feature is hard to
learn, and it **disappears when the complex feature is learnable**, where reliance
instead tracks predictive strength. The most parsimonious reading of our data is
that the observed "simplicity bias" is entangled with — and in this setting
largely explained by — the optimisation difficulty of the more complex feature.
We do not claim to have disproved the claim; we claim to have found a boundary.

---

## 1. Introduction

Neural networks can represent extremely complicated functions, yet standard
training procedures are repeatedly reported to latch onto simple predictive
patterns and ignore more complex ones that are equally or even more useful
(Shah et al., 2020; Morwani et al., 2023). This "simplicity bias" (SB) is often
invoked to explain non-robustness, shortcut learning, and failures under
distribution shift.

The claim is seductive precisely because "simple" feels self-evident — until one
tries to say what it means. Shah et al. (2020) frame it as piecewise-linear
features with few pieces; Morwani et al. (2023) define it as *low-dimensional
input dependence*; Rahaman et al. (2019) study a different but related bias
toward low-frequency functions. These are not the same thing, and a study that
shows a network "prefers simple features" is only as strong as the definition it
used and the controls it ran.

The challenge is explicitly not to reproduce the claim. It is to ask *under what
conditions the claim fails, and how we would know scientifically*. The most
dangerous confound is stated in the brief itself: perhaps the network prefers the
simple feature **not because it is simple, but because it is easier to optimise**.
Our study is designed around exactly this confound.

The contribution of this proposal is a synthetic design in which (a) the *only*
difference between the two features is their functional complexity, (b) their
predictive strengths are set analytically and matched exactly when desired, and
(c) the complex feature's isolated learnability is measured directly, so that
"prefers simple" and "can only learn simple" can be told apart.

## 2. Literature Review

**Shah, Tamuly, Raghunathan, Jain & Netrapalli (2020), "The Pitfalls of Simplicity
Bias in Neural Networks," NeurIPS 33, arXiv:2006.07710.** The primary reference.
The authors introduce piecewise-linear and image-based datasets with multiple
predictive features of *varying simplicity and predictive power*, and argue SB is
*extreme*: SGD-trained networks can "exclusively rely on the simplest feature and
remain invariant to all predictive complex features." Four observations are made,
two of which directly motivate our design: (i) reliance on the simplest feature;
(iii) SB **persists even when the simplest feature has less predictive power**
than the complex ones; (iv) ensembles and adversarial training do not mitigate
it. Notably, the authors themselves stress that "the precise notion of simplicity
remains vague." Our study takes their claim, fixes a precise notion of simplicity,
and probes observation (iii) while adding the missing control: the complex
feature's *measured* learnability.

**Morwani, Batra, Jain & Netrapalli (2023), "Simplicity Bias in 1-Hidden Layer
Neural Networks," NeurIPS 36, arXiv:2302.00457.** Gives a rigorous
characterisation for one-hidden-layer networks in the infinite-width limit:
SB is identified with **low-dimensional input dependence** (LD-SB) and proved for
linearly separable data in an *independent features model*, where the network
primarily depends on the linearly-separable (1-D) subspace even with many other
complex features present. Two things matter for us. First, the analytical setting
resembles our construction (independent features, a simple separable direction, a
harder one). Second, their result is about the *lazy/rich regime and scale of
initialisation* — i.e., it is an optimisation-and-architecture statement as much
as a data statement, reinforcing that "simplicity" and "ease of optimisation" are
hard to separate.

**Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio & Courville (2019),
"On the Spectral Bias of Neural Networks," ICML, PMLR 97:5301–5310.** Provides the
low-frequency notion of simplicity: networks fit lower-frequency components of a
target function first. The authors also show the geometry of the data manifold
matters. This is the definition our "complexity = number of decision boundaries /
frequency of the feature→label map" operationalisation is closest to.

**Open questions the literature leaves for a sceptic.** (1) When a preference for
the simple feature is observed, how much of it is *simplicity* versus differential
*learnability*? (2) Is the preference robust when predictive strength is
independently varied at *matched* complexity, and vice versa? (3) Does the
preference survive a direct positional control? (4) Does it change with training
budget, learning rate, or architecture? Our study addresses (1)–(4) in a minimal
setting.

## 3. Research Question

**RQ.** *When two predictive features are equally distributed and matched in
predictive strength but differ in functional complexity, does a small MLP
preferentially rely on the simpler one — and if so, is the preference attributable
to complexity itself, or to the complex feature being harder to learn (optimise)?*

Sub-questions:
- **RQ-a (strength):** When the complex feature is made *more* predictive, does
  reliance shift toward it, or does the simple feature still win?
- **RQ-b (learnability):** Is there a complexity level at which the complex
  feature is learnable in isolation, and does the preference track that boundary?
- **RQ-c (optimisation):** Do training budget, learning rate, or architecture
  change which feature the network ultimately relies on?

## 4. Hypothesis and Null Hypothesis

**H1 (simplicity preference).** At matched predictive strength, the network will
rely substantially more on the simple feature than on the complex feature, as
measured on examples where the two disagree — i.e. the preference index
P (defined in Section 6) will be strongly positive. *Rationale:* this is the
literature claim (Shah et al., observation i) under a matched-strength control.

**H2 (strength dominates when both are learnable).** Where the complex feature is
learnable in isolation, changing predictive strength will flip reliance in the
direction of the stronger feature, i.e. P will follow α vs β rather than
complexity. *Rationale:* if reliance tracks strength where the complex feature is
learnable, then "complexity" is not the operative variable there.

**H3 (learnability, not simplicity, is the operative variable).** The magnitude of
the preference will co-vary with the complex feature's *isolated* learnability,
rather than with complexity per se: the preference should be large exactly where
the complex feature is hard to learn alone. *Rationale:* the brief's central
warning — a feature may be preferred because it is easier to optimise.

**H0 (null).** Complexity has no systematic effect once predictive strength and
the feature distributions are matched: the network shows no reliable preference
for either feature across seeds (P centred on 0), and any apparent preference is
accounted for by predictive strength or by optimisation conditions.

**Falsifiability (stated before data).** Each hypothesis makes a prediction that
could be contradicted: H1 predicts P>0 at matched strength; H2 predicts a sign
flip of P when strength is flipped *only where the complex feature is learnable*;
H3 predicts that the preference vanishes at low complexity where the complex
feature is learnable. A finding of P≈0 at matched strength would falsify H1; a
finding that P stays strongly positive even when the complex feature is learnable
and stronger would falsify H2/H3 and support an intrinsic simplicity preference.

**Alternative explanation (required by the brief).** Perhaps the network does not
select the simple feature *because it is simple*. Perhaps the complex feature is
simply **harder to optimise** with the given architecture, initialisation,
optimiser and budget — it may be effectively unlearnable, so the network defaults
to the only feature it *can* learn. A second alternative: the preference could be
an artefact of **input stacking order** (the optimiser/architecture treating column
0 specially). A third: it could be **insufficient training** and vanish with a
longer budget. Sections 5–6 specify the controls that separate these.

## 5. Proposed Experiment Suite

### 5.1 Dataset construction (synthetic, fully controlled)

Inputs are two independent columns `x = (x_s, x_c)`, each Uniform(−1, 1).
N = 10,000 train, 2,000 validation, 10,000 test, plus a 20,000-example
"conflict" evaluation set (defined below). Two balanced binary features are
constructed:

| | definition | decision boundaries | marginal |
|---|---|---|---|
| simple | `u_s = sign(x_s)` | **1** (at x=0) | 50/50 |
| complex | `u_c = sign(sin(kπ·x_c))` | **2k** | 50/50, identical input marginal |

`k ∈ {1, 2, 4, 8, 32}` is the complexity knob. Because both features are balanced
and share the same uniform input marginal, changing `k` changes **only** the
functional complexity of the `x_c → label` map.

The label comes from a noisy threshold vote:

```
y = +1  iff  α·u_s + β·u_c + ε ≥ 0,     ε ~ N(0, 1)
```

with fixed noise σ=1. This is deliberately the same *multiplicative/additive
feature-combination* structure used by Shah et al., and the two features are
**non-combinable**: on the half of examples where `u_s ≠ u_c` the signal cancels,
so no classifier can exceed chance there. Consequences that make the design
controllable:

- The single-feature Bayes accuracies are **exact closed forms**:
  `A_s = ½Φ(α+β) + ½Φ(α−β)` and `A_c = ½Φ(α+β) + ½Φ(β−α)`.
- The features are matched in predictive strength **iff α = β** (then
  `A_s = A_c = ½Φ(2) + ¼ = 0.7386`). This is what turns "matched strength" from a
  hope into a calculation.
- At α = β, neither feature is statistically favoured: each alone is as good as
  the other, and using both is no better than using one. Any systematic reliance
  is therefore attributable to the network, not to the task.

**Why k = 1 is a null-calibration anchor.** On x ∈ (−1,1),
`sign(sin(π·x)) ≡ sign(x)`. So at k=1 the "complex" feature is *mathematically
identical* to the simple one. Any systematic preference at k=1 can only be a
positional/initialisation artefact; the real effect must show up as a departure
from this anchor. (A direct positional control, swapping the columns, is added in
Section 6.)

**Why the features are treated as genuinely predictive.** Rather than assuming the
simple feature is easier, we *measure* it: an isolated task (`y` depends on only
one feature) is trained for each feature and complexity level, and its test
accuracy is compared with the analytic Bayes value 0.8413 = Φ(1). "Learnable" then
has an operational meaning: near-Bayes isolated accuracy.

### 5.2 Experimental conditions (each isolates one variable)

Everything except the named variable stays fixed within a comparison
(architecture, optimiser, LR, batch size, budget, seeds, data stream).

- **A. Baseline (matched strength).** α=β=1, k=8. Tests H1 with strength matched.
- **B. Predictive-strength control.** α=1.3, β=0.7 and α=0.7, β=1.3 (total signal
  held at α+β=2), at **k=8** (complex hard) *and* **k=2** (complex learnable).
  Tests H2/RQ-a and, by comparing the two k values, H3.
- **C. Complexity sweep, matched strength.** k ∈ {1, 2, 4, 8, 32}, α=β=1. Traces
  the preference against complexity.
- **D. Optimisation-difficulty controls.**
  - *D1 (isolation learnability):* `y` depends on the simple feature only, or on
    the complex feature only, at each k ∈ {2,4,8,32}. This is the key alternative-
    explanation experiment. It is run as a gradient of conditions.
  - *D2 (budget):* baseline retrained for 5× epochs (500) — does the preference
    fade if the complex feature is simply slow?
  - *D3 (learning rate):* baseline at LR×0.3 and LR×3.
- **E. Robustness.** A second architecture (2→16→16→1, ReLU) at baseline; plus the
  seed variation that is present in every condition (5 seeds).
- **F. Sanity / control conditions.** (i) shuffled-label control (labels permuted:
  no signal should give no preference and ~50% test accuracy); (ii) positional
  control (baseline with the two input columns swapped); (iii) a 4×-data version
  of the hardest joint condition (k=32) and of the k=8 isolation task, to separate
  "complex" from "data-starved per region".

**Conditions deliberately *not* included (and why).** We do *not* sweep width/depth
extensively, σ (label noise), or the correlation between features: with 5 seeds
these would be under-powered and would not isolate the simplicity-vs-learnability
question. We note them as future work (Section 9).

### 5.3 Model architecture and training procedure

- **Model:** MLP 2→64→64→1, `tanh` (robustness arm: 2→16→16→1, ReLU). Deliberately
  tiny and conventional; no capacity tricks.
- **Loss / optimiser:** binary cross-entropy with logits; Adam, LR 1e-3, batch 256,
  weight decay 0, **no early stopping** (so that *stopping* is never a
  condition-dependent variable).
- **Budget:** 100 epochs (500 for D2). Fixed within every comparison.
- **Initialisation:** Kaiming-uniform weights, zero biases, seeded.
- **Data order:** a dedicated seeded PyTorch generator, reshuffled each epoch.

### 5.4 Evaluation methodology (metrics fixed before running)

1. **Conflict preference index P (primary).** On the "conflict" set (`u_s ≠ u_c`
   by construction), `P = frac(predictions = u_s) − frac(predictions = u_c)`,
   in [−1, 1]. `P > 0` means the model follows the **simple** feature; `P < 0` the
   **complex** one. This is **label-free** — it asks which feature the model
   follows, not which is correct (on conflicts at α=β the true label is pure
   noise, so accuracy there is uninformative by construction).
2. **Permutation-ablation reliance.** `drop_s = acc(X) − acc(X with column s
   permuted)`; likewise `drop_c`. This is clean here because the two columns are
   independent *by construction*, so permuting one preserves the input joint
   distribution and only destroys that feature's information.
3. **Whole-run accuracy** vs the analytic Bayes value.
4. **Learning curves**: P and accuracy evaluated every few epochs (fixed schedule,
   fixed evaluation data) to see *when* reliance is established.

### 5.5 Reproducibility and software

Deterministic seeds throughout (NumPy `SeedSequence` child streams for inputs,
label noise, conflict set, evaluation permutation; separate PyTorch seeds for
initialisation and data order). Paired conditions share their data and
initialisation seed, so cross-condition comparisons are paired. Every run is saved
to `results/runs/*.json` with its full settings and metrics; a `run_manifest.json`
records software versions. Requires only NumPy, SciPy, scikit-learn, PyTorch (CPU)
and Matplotlib (see `requirements.txt`). Exact commands are in `README.md`.

## 6. Controls and Confounding Variables

| Control / baseline | What it can establish | What it cannot establish |
|---|---|---|
| **Matched predictive strength (α=β)** | That any preference is not due to the complex feature being weaker — strengths are equal *by construction* (both Bayes accuracies = 0.7386). | Whether an *intrinsic* preference exists independent of *learnability* (needs D1). |
| **k=1 null anchor** | That the metric does not automatically report "simple" — when the features are mathematically identical the preference must be ~0. | Nothing about higher complexity. |
| **Positional (column-swap) control** | That the preference is for the simple *feature*, not for *column 0*. | Whether the effect generalises beyond 2 features. |
| **Isolated learnability gradient (D1)** | Whether the complex feature is *learnable at all* in isolation, and how that changes with k — the crux of the alternative explanation. | Whether a learnable-but-unpreferred feature would be learned *if forced* (e.g. by masking the simple one). |
| **Strength flip at k=8 vs k=2 (B)** | Whether reliance follows *strength* where the complex feature is learnable (H2). | Causality beyond the tested ranges. |
| **Budget (D2) / LR (D3)** | Whether the preference is a finite-training or step-size artefact. | Exhaustive optimiser choices. |
| **Second architecture (E)** | Whether the preference is architecture-specific within a small family. | Whether large/conv/transformer models behave the same. |
| **Shuffled labels (F-i)** | That the pipeline yields ~chance accuracy and no *consistent* preference with no signal (guards against a metric bug). | — |
| **Logistic-regression baseline** | What "simplicity" looks like when *imposed by model class*: a linear model can only use the simple, linearly-separable feature. | Anything about gradient-trained MLP preferences. |
| **Analytic oracle** | The exact Bayes accuracies and the true predictive contribution of each feature. | Real-data behaviour. |

**Confounders we explicitly acknowledge and quantify.** (i) *Complexity =
frequency = number of regions = samples-per-region*: increasing k changes several
things at once; we quantify samples-per-region (at k=32, 10k points give ≈156 per
stripe) and run a 4×-data control to test the data-coverage explanation. (ii)
*The features are non-combinable* in this vote model, so the task is effectively
"pick a feature"; this is the same structure as Shah et al. but limits
generalisation to tasks with feature *synergy*. (iii) *Computational click*: all
results are relative to this model family, optimiser and budget. The boundaries we
report are boundaries of behaviour, not of mathematics.

## 7. Expected Outcomes and Interpretation (pre-registered)

Decided **before** running; the implemented results in Section 8 are kept separate.

- **Outcome 1 — simple feature consistently preferred (matched strength).**
  Supports the original claim (H1). Preferentially on the record as an *extreme*
  result if the ablation drop on the complex feature is ~0 and on the simple
  feature ~0.24 (= the whole simple-feature contribution).
- **Outcome 2 — reliance follows *strength* where the complex feature is
  learnable.** With complex stronger, P flips negative at k=2. Supports H2 and, if
  the flip vanishes at k=8, weakens the claim to a *learnability* effect (H3).
- **Outcome 3 — preference changes with optimisation conditions** (budget, LR,
  architecture). Suggests the effect is optimisation-dependent, qualifying the
  claim; a shift toward the complex feature with more training would substantially
  weaken H1.
- **Outcome 4 — inconsistent preferences across seeds/architectures.** If P has
  large seed variance, any single-run conclusion is unsafe; this would make the
  claim hard to test in small experiments and is itself an important negative
  result.
- **Outcome 5 — preference tracks isolated learnability, not complexity.** The
  preference is large exactly where the complex feature is unlearnable in
  isolation, and ~0 where it is learnable. This would most strongly support the
  alternative explanation and *qualify* (not disprove) the claim.
- **Outcome 6 — an intrinsic preference survives everything** (P stays strongly
  positive even where the complex feature is learnable and stronger). Would
  *support* an intrinsic simplicity preference and be evidence against the pure
  optimisation explanation.

For each, the "additional experiment that would help" is noted in Section 8 where
relevant. Pre-registered decision rule: the claim is treated as *qualified* if
Outcome 5 holds (preference co-varies with isolated learnability); it is treated
as *weakened* only if a learnable, equally- or more-predictive complex feature is
still systematically ignored with adequate controls.

## 8. Optional Experimental Results

*Implemented and run. Reported separately from Section 7. Every number below is
produced by the code in `src/` and stored in `results/`; regenerate with the
commands in `README.md`.*

### 8.1 Methodology

Settings exactly as Section 5. 5 seeds per condition (seeds are the unit of
replication; epochs are never treated as independent). "Preference P" is the
conflict metric of Section 5.4; "acc" is test accuracy; "isolated complex acc" is
the test accuracy of a model trained on a label that depends only on the complex
feature (Bayes = 0.8413). Full table in `results/analysis_summary.md`; figures in
`results/figures/`.

### 8.2 Results

**Table 1 — complexity sweep at matched strength (α=β=1).** k=1 is the null anchor.

| condition | k | preference P | test acc | drop_simple | drop_complex |
|---|---|---|---|---|---|
| k1_null | 1 | −0.321 ± 0.327 | 0.737 | 0.083 | 0.153 |
| k2 | 2 | +0.022 ± 0.187 | 0.740 | 0.125 | 0.119 |
| k4 | 4 | +0.909 ± 0.051 | 0.741 | 0.236 | 0.009 |
| **baseline** | 8 | **+0.990 ± 0.004** | 0.737 | 0.242 | 0.000 |
| k32 | 32 | +0.993 ± 0.003 | 0.742 | 0.243 | −0.001 |

(Bayes, matched: 0.7386.) The preference is essentially *on or off*: near-total for
k≥4, absent for k≤2. The k=1 anchor shows no systematic preference — the metric is
not hard-wired to answer "simple". `drop_complex ≈ 0` at k=8/32 means permuting the
complex column costs the model *nothing*: it never used that feature.

**Table 2 — isolated learnability of the complex feature (label depends only on it;
Bayes 0.8413).** Time budget 100 epochs unless noted; `n=10k` unless noted.

| k | isolated complex acc | interpretation |
|---|---|---|
| 2 | 0.825 ± 0.006 | learnable (near Bayes) |
| 4 | 0.699 ± 0.038 | partially learnable |
| 8 | 0.520 ± 0.007 | ~unlearnable |
| 32 | 0.502 ± 0.007 | not learnable |
| 8, 5× epochs | 0.590 ± 0.039 | still poor with more time |
| 8, 4× data | 0.550 ± 0.040 | still poor with more data |
| 8, ReLU 16×16 | 0.530 ± 0.005 | still poor with another architecture |

Simple-only isolated acc: 0.843 ± 0.002 (matches Bayes). The complex feature's
learnability collapses between k=2 and k=8, and the collapse is *not* fixed by
5× training, 4× data, or a different small architecture.

**Table 3 — the preference appears exactly where learnability collapses.**
Combining Tables 1–2, the joint preference tracks isolated learnability:
k=2 (learnable, 0.83) → P≈0.02; k=4 (partial, 0.70) → P≈0.91;
k=8/32 (unlearnable) → P≈0.99.

**Table 4 — predictive-strength control, at k=8 (unlearnable) vs k=2 (learnable).**

| (α, β) | k=8 preference P | k=8 acc | k=2 preference P | k=2 acc |
|---|---|---|---|---|
| simple stronger (1.3, 0.7) | +0.996 | 0.849 | +0.977 | 0.851 |
| matched (1.0, 1.0) | +0.990 | 0.737 | +0.022 | 0.740 |
| complex stronger (0.7, 1.3) | **+0.895** | **0.627** | **−0.745** | 0.821 |

At **k=8** the network *still* follows the simple feature even when the complex
feature is much more predictive: its accuracy (0.627) matches the *weaker*
simple-feature Bayes accuracy (0.626) and leaves the attainable 0.852 on the
table — replicating Shah et al.'s observation (iii). At **k=2**, where the complex
feature is learnable, reliance *flips to the stronger feature* (P = −0.745). So
strength governs reliance only where the complex feature is learnable.

**Table 5 — other controls.**

| condition | result | reading |
|---|---|---|
| positional (columns swapped) | P = **−0.990** ± 0.004, drop_complex 0.235 | follows the simple *feature*, not column 0 |
| architecture (ReLU 16×16), baseline | P = +0.984 | not specific to tanh/width-64 |
| LR ×0.3 / ×3 | P = +0.987 / +0.987 | not a step-size artefact |
| 5× epochs (k=8) | P = +0.929 ± 0.065 | preference does *not* fade with more time |
| 4× data (k=32) | P = +0.998 | not a data-coverage artefact |
| shuffled labels | acc 0.513 ± 0.080, P +0.127 ± 0.329 | ~chance accuracy, no *consistent* preference |
| logistic regression (k=8) | acc 0.727, P +0.929 | a *linear* model is forced onto the simple feature |

**Learning dynamics.** In every baseline seed, the preference is established in
the *first epoch* (P ≈ 0.98 at epoch 1) and never changes: the network finds the
simple feature immediately and never recovers the complex one. See
`results/figures/fig_learning_curves.png` and `fig_preference_and_learnability_vs_k.png`.

### 8.3 Analysis

The implemented experiment gives a coherent, and somewhat *qualifying*, answer.

1. **The claim is supported under matched strength (Outcome 1).** At k=8, P=0.990
   across 5/5 seeds with tiny variance, and the complex feature is completely
   unused (drop_complex≈0). A naive report would stop here.
2. **But the preference is conditional, and it tracks learnability (Outcome 5).**
   Where the complex feature is learnable (k=2, isolated acc 0.83), there is *no*
   preference (P≈0.02) and reliance is distributed (drop_s≈drop_c). Where it is
   unlearnable (k≥8), the preference is total. The switch happens right where the
   isolated learnability collapses.
3. **Strength matters only where the complex feature is learnable (Outcome 2).**
   At k=2 the network follows the stronger feature even when that is the complex
   one (P=−0.745); at k=8 it follows the simple one even when the complex one is
   stronger (P=+0.895, accuracy pinned to the *weaker* feature's Bayes value). This
   last result is a faithful replication of Shah et al. (observation iii) — and
   our control shows *why*: at k=8 the more predictive feature is effectively
   unlearnable, so "prefers simple" and "can only learn simple" coincide.
4. **It is not an artefact** of the metric (k=1 anchor), of column order (swap),
   of step size (LR), of budget (5× epochs), of data coverage (4× data), or of the
   specific small architecture (ReLU arm). Seed variance is tiny at k≥4.

**Honest nuance — the boundary is not perfectly clean.** At k=4 the complex feature
is *partially* learnable in isolation (0.70), yet the joint preference is already
strong (+0.91). So the operative factor is not a simple "learnable vs unlearnable"
dichotomy: when an easy feature competes during training, it appears to win the
optimisation race and starve the harder (but learnable) feature. That is still an
*optimisation* explanation rather than an intrinsic-simplicity one, but it means
we cannot claim a sharp learnability threshold. We also cannot rule out a genuine,
small intrinsic simplicity preference layered on top of the optimisation effect.

**What this does and does not say.** Under our operationalisation and setup, H1
holds strongly at high complexity but is **not** attributable to complexity
independently of learnability; H2 holds where the complex feature is learnable;
H3 is supported (the preference co-varies with isolated learnability). We therefore
**qualify** the claim rather than disprove it: the observed "simplicity bias" is
inseparable, in this setting, from the optimisation difficulty of the more complex
feature. A future experiment that would sharpen this: force the network onto the
complex feature (e.g. mask `x_s`) and ask whether a *learnable* complex feature is
then used at all, or jointly optimise with a warm-up / feature-widening scheme to
see if the easy feature's head-start is the whole story.

## 9. Limitations

- **One operationalisation of simplicity.** We measure "number of decision
  boundaries / frequency". Other notions (linear separability, description length,
  low dimensionality in Morwani et al.'s sense) may behave differently; our
  conclusions are about *this* definition.
- **Non-combinable features.** The vote model gives no synergy, so the task is
  "pick a feature". Real tasks often reward combining features; the competition
  dynamics we observe may differ when features are complementary.
- **Confounded complexity knobs.** Increasing k changes frequency, number of
  regions, and samples-per-region simultaneously. We bounded the data-coverage
  explanation with a 4×-data control, but the knobs remain partly entangled.
- **Small-scale, single model family.** A tiny MLP, Adam, one LR grid, 5 seeds.
  The boundary between "learnable" and "not" is a property of
  (feature, architecture, optimiser, budget) jointly, not of the feature alone;
  a CNN/transformer or a warm-up schedule might move it.
- **Noisy metrics under the null.** The shuffled-label and k=1 conditions show
  non-trivial seed variance, a reminder that small per-seed differences are not
  evidence; we rely on effect sizes that exceed this null spread.
- **Cannot establish causality about real networks.** Conclusions are about this
  controlled synthetic setting; transferring to large-scale vision/NLP models is
  an open empirical question.
- **The k=1 anchor is indirect** (k=1 makes the features identical, not merely
  equally complex); the column-swap control is the direct counterpart, and it
  agrees.

## 10. Conclusion

We set out to test — and, if possible, to falsify — the claim that neural networks
preferentially learn simpler predictive features, with a design that separates
feature *complexity* from predictive *strength* and, critically, from *learnability*.
Using a synthetic two-feature task with analytically matched strength and a
label-free reliance metric, and running the experiment with paired seeds, we find:

- the claim **holds** at matched strength when the complex feature is hard to
  learn (P≈0.99, complex feature unused); and it holds *even when* the complex
  feature is more predictive — a faithful replication of the headline effect;
- but the preference **vanishes** (P≈0) at a complexity level where the complex
  feature is learnable, and reliance there instead **follows predictive strength**;
- and the preference is not an artefact of the metric, column order, learning
  rate, training budget, data size, or the specific small architecture.

The most parsimonious reading of our evidence is that, in this setting, the
observed simplicity bias is largely **entangled with the optimisation difficulty
of the more complex feature**. We did **not** disprove the claim: a genuine,
intrinsic preference may still exist, and our complexity definition is only one of
several. But we did map a boundary — the claim's strength depends on whether the
complex feature is *learnable*, not merely on whether it is *complex* — and that is
exactly the kind of qualification the challenge is designed to elicit.

## 11. References

1. Shah, H., Tamuly, K., Raghunathan, A., Jain, P., & Netrapalli, P. (2020).
   *The Pitfalls of Simplicity Bias in Neural Networks.* Advances in Neural
   Information Processing Systems 33 (NeurIPS 2020). arXiv:2006.07710.
2. Morwani, D., Batra, J., Jain, P., & Netrapalli, P. (2023). *Simplicity Bias in
   1-Hidden Layer Neural Networks.* Advances in Neural Information Processing
   Systems 36 (NeurIPS 2023). arXiv:2302.00457.
3. Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F.,
   Bengio, Y., & Courville, A. (2019). *On the Spectral Bias of Neural Networks.*
   Proceedings of the 36th International Conference on Machine Learning (ICML),
   PMLR 97:5301–5310. arXiv:1806.08734.

(Citation details verified against the publishers' pages; see `references.md`.)

---

## Appendix A — Decisions for you to own before submitting

This appendix is *not* part of the proposal proper. It lists choices where your
own judgement should override the defaults above, and it flags places where the
document state is the design's view rather than your personal view.

1. **The operational definition of simplicity (Section 5.1).** I chose "number of
   decision boundaries / frequency". If your instinct is that simplicity should
   mean, e.g., *linear separability* or *low dimensionality* (Morwani et al.), say
   so in the interview — and note that the design could be re-run with those
   definitions largely unchanged (they change what `u_c` is, not the machinery).
2. **The value of k and the strength pairs.** k ∈ {1,2,4,8,32}, and (1.3, 0.7).
   These were chosen so that the learnability transition sits inside the swept
   range; justify or adjust them.
3. **The primary metric.** P (conflict preference) is primary because it is
   label-free and distribution-preserving. Ablation is secondary. If you prefer
   to foreground ablation (or an accuracy-based measure), that is a defensible
   change — but then the argument must handle the fact that ablation is confounded
   by redundancy when a model uses both features.
4. **Scope of the claim you are willing to defend.** The write-up claims a
   *qualification*, not a falsification. If you think the evidence supports a
   stronger statement, we should strengthen the controls (e.g. masked-feature
   forcing, warm-up schedules) rather than the wording.
5. **Sections 7 vs 8.** Section 7 is the design's *predictions*; Section 8 is what
   *was observed*. Keep them separate when you present. Do not describe Section 8's
   numbers as if they had been predicted.
6. **Framing of the "logistic regression" baseline.** It currently appears as a
   "simplicity-by-model-class" reference. If you consider it unnecessary (the
   brief warns against baselines that do not answer the question), it can be
   demoted to a footnote — but it does usefully show what "forced" simplicity
   looks like.
