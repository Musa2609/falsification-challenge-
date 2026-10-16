# Interview Questions and Explanatory Answers 

## 1. What exactly are you trying to falsify?

**Question:** What exactly are you trying to falsify? 

**Expected answer:** The broad claim that "when multiple features can be used to solve a task, neural networks tend to preferentially learn simpler predictive features over more complex ones" (Shah et al., 2020; Morwani et al., 2023). 

**Explanation:** Our objective is not to "disprove" the claim wholesale, but to test its boundaries. The most critical test for a scientific claim is the falsification mindset: what conditions would make it fail? In particular, we want to separate "prefers simple because it is simpler" from "prefers simple because the complex feature is harder to learn (optimize)". 

**Evidence:** Our design tests (a) matched predictive strength (α=β), (b) strength reversal (flip α,β), (c) the complex feature's *isolated learnability* as a function of complexity k (number of decision boundaries). 

## 2. What does "simple" mean in your study?

**Question:** What does "simple" mean in your study?

**Expected answer:** We operationalize simplicity as **the number of decision boundaries of the feature→label function along that feature's input axis**. 

**Explanation:** `u_s = sign(x_s)` has 1 boundary; `u_c = sign(sin(kπ·x_c))` has 2k boundaries. This is a precise, measurable operationalization. It is close to the low-frequency/spectral bias view (Rahaman et al., 2019) and to the "few pieces" of piecewise-linear features in Shah et al. (2020). We note other notions exist (e.g. linear separability, low-dimensional input dependence, description length) and we treat this as *one* operationalization — hence the qualifier "under this operationalization" throughout.

**Caveat:** Increasing k also changes the number of regions and samples-per-region; we controlled for data coverage (4× data) and for input marginals (identical U(−1,1)) to reduce confounding.

## 3. Why did you choose synthetic data?

**Question:** Why did you choose synthetic data?

**Expected answer:** Because we needed to **independently control and measure** the two variables that are easy to confound in real data: (1) the *complexity* of the feature→label map and (2) the *predictive strength* of each feature. 

**Explanation:** In natural datasets the features are not labeled and their strengths are usually not known analytically. Our construction gives:
- identical input marginals for both features
- closed-form single-feature Bayes accuracies: `A_s = ½Φ(α+β)+½Φ(α−β)`, `A_c = A_s(β,α)`
- exactly matched strength iff α=β
- k=1 as a mathematical null anchor (features identical)
- perfect isolation of k vs everything else

So we traded "realistic-looking" for "controllable". The tradeoff is explicit in our Limitations.

## 4. How do you separate simplicity from predictive strength?

**Question:** How do you separate simplicity from predictive strength?

**Expected answer:** Two ways.

1. **Analytic control.** We set coefficients (α,β) in `y = 1[α·u_s + β·u_c + ε≥0]`. Single-feature Bayes accuracies are exactly `A_s(α,β), A_c(α,β)=A_s(β,α)`, so we can (a) match them (α=β), and (b) flip which is stronger (α<β and α>β) while keeping other factors fixed.
2. **Explicit measurement.** For each configuration we also measure (a) whole-test accuracy, (b) **conflict preference index P** on examples where the two features disagree (`u_s ≠ u_c`), and (c) **permutation ablation drops** (permute a column => only that feature's information is destroyed, since columns are independent by construction).

**Key metric:** The **conflict preference index P** is label-free: it tells us *which feature the model follows*, not which is correct (on conflicts at α=β the label is pure noise). So when α=β, following either is equally accurate in expectation — any systematic preference is due to the model, not to the task.

**Actual evidence:** At k=8 and matched strength, P ≈ +0.99 across 5 seeds → strong preference for the simple feature. At k=2 and matched strength, P ≈ +0.02 → no preference. At k=2 and α<β (complex stronger), P ≈ −0.75 → follows the stronger (complex) feature. At k=8 and α<β (complex stronger), P ≈ +0.90 → still follows simple, and accuracy is pinned to the weaker-feature Bayes value. So strength matters *only* when the complex feature is learnable.

## 5. How do you assess optimization difficulty?

**Question:** How do you assess optimization difficulty?

**Expected answer:** We measure **isolation learnability** directly. That is, we train models on tasks where the label depends on *only* one feature: `y = 1[α·u_s + ε>0]` and `y = 1[β·u_c(k) + ε>0]`, with the same architecture, optimizer, LR, budget, and seeds.

**Explanation:** 
- Simple-only (strength 1): test accuracy ≈ 0.843 (Bayes = 0.8413) → learnable.
- Complex-only, k=2: 0.825 → learnable.
- Complex-only, k=4: 0.699 → partially learnable.
- Complex-only, k=8: 0.520 → ~unlearnable (chance = 0.5).
- Complex-only, k=32: 0.502 → not learnable.
- k=8 with 5× epochs: 0.590 → still poor.
- k=8 with 4× data: 0.550 → still poor.
- k=8 with ReLU 16×16: 0.530 → still poor.

Also, **learning curves** show the preference index at multiple epochs: in the joint k=8 baseline, P was already ≈ 0.98 at epoch 1 and stayed high — the simple feature "wins" the optimization race and is never relinquished, while the complex feature is effectively unlearnable. Budget (5× epochs) did not flip it. 

**Why this matters:** The "simplicity bias" claim must be qualified if preference co-varies with isolation learnability. That is exactly what we observe: preference disappears when both are learnable (k=2), appears when the complex feature becomes hard to learn (k≥4), and is extreme when unlearnable (k≥8). 

## 6. Why did you choose this architecture?

**Question:** Why did you choose this architecture?

**Expected answer:** A **tiny, conventional MLP**: 2 → 64 → 64 → 1, tanh. 

**Explanation:** 
- The task is minimal: 2-D input. Capacity is more than enough to learn either feature in isolation (simple-only and complex-only k=2 reach near-Bayes). This avoids over-engineering and keeps compute trivial (CPU-friendly).
- We deliberately **don't** use a large network. If an effect disappears at scale that's worth knowing; but our question is about whether a *standard* small network exhibits the reported behavior. The brief explicitly prefers the smallest experiment that can answer the question.
- We included a second arm: 2 → 16 → 16 → 1, ReLU at baseline — the preference was similar (P = +0.984). So the result isn't specific to width 64/tanh.

**Tradeoff:** With this arch the complex feature at k≥8 is unlearnable in isolation; we show this is *robust* to 5× data, 5× epochs, and a different small architecture. We explicitly don't claim generality to CNNs/transformers (listed in Limitations).

## 7. How do you measure which feature the model learned?

**Question:** How do you measure which feature the model learned?

**Expected answer:** **Primarily the conflict preference index P**, computed on the "conflict test set" where `u_s ≠ u_c` (separately generated, evaluation-only). Secondary: **permutation ablation drops**.

**Details:**
- P = `frac(pred = u_s) − frac(pred = u_c)`, in [−1,1]. Values near +1 mean the model follows the simple feature; near −1 means it follows the complex feature; near 0 means no systematic preference (distributed reliance).
- Because `u_s ≠ u_c` and α=β, the true label is noise on these points — so the metric is not measuring correctness, it's measuring **which feature the network's function depends on**. This is exactly what "which feature did the model learn to rely on?" means.
- Permutation ablation: permuting column 0 destroys only the simple feature's information (and does nothing else because columns are independent). `drop_simple = acc(X) − acc(X with col0 permuted)`. At k=8 matched, `drop_simple ≈ 0.242`, `drop_complex ≈ 0.000` — consistent with exclusive reliance on the simple feature.

**Sanity checks:** on an oracle using only `u_s`, P = +1 and drop_complex ≈ 0; on an oracle using only `u_c`, P = −1 and drop_simple ≈ 0; shuffled-label control gives ~0 preference and high seed variance.

## 8. What result would weaken your hypothesis?

**Question:** What result would weaken your hypothesis?

**Expected answer:** Our main hypothesis (H1 at matched strength, high k) is: the network relies substantially more on the simple feature. It would be **weakened** if:
- at α=β and k=2 (both learnable) the network still preferred the simple feature strongly (P >> 0) instead of being near 0;
- at α<β and k=2 (complex stronger and learnable), the network still followed the simple feature (P > 0) instead of flipping to negative (following the stronger feature);
- at α=β and k=1 (identical features), the network showed a consistent preference for one column (systematic bias) — this would indicate a positional artifact;
- with more training budget (5× epochs) at k=8, preference shifted toward the complex feature.

**Actual result:** H1 is supported at k≥4, but the "simplicity per se" explanation is weakened because: at k=2 (both learnable) preference vanishes (P≈0) and tracks strength (P flips sign when complex stronger). That means when both features are equally "optimizable", no simplicity preference appears — the preference is conditional on the complex feature being hard to learn. This *qualifies* rather than disproves the claim.

## 9. What confounders remain?

**Question:** What confounders remain?

**Expected answer:**

- **Multiple aspects of complexity.** Increasing k changes the number of decision boundaries, the number of regions, and the number of samples per region (≈ n/(2k)). We addressed samples-per-region with a 4×-data control (k=32, 4× data → preference stayed extreme). Other aspects remain partially entangled.
- **Operationalization of "simple".** We only test one definition (number of decision boundaries / frequency). The literature offers others (linear separability, low-dimensional input dependence) which might behave differently. Our conclusion is "under this operationalization".
- **The vote model is non-combinable.** At α=β, features are redundant in a particular way: neither feature helps on conflicts and there's no synergy. This matches Shah et al.'s style, but real tasks often reward combining features; competition dynamics could differ.
- **Scale and architecture.** We used a tiny MLP (2×64×64×1, tanh) and one small alternative (ReLU 16×16). The "learnability boundary" (between k=2 and k=4) is a property of *(feature, architecture, optimizer, budget)*, not of the feature alone. A CNN or transformer might move it.
- **Seed variance under the null.** Shuffled-label and k=1 conditions show substantial seed-to-seed variance (std ≈ 0.33 for P). This means we cannot reliably detect tiny preferences with n=5; we only interpret effects that clearly exceed this null noise level (e.g. P≈+0.99 vs null ~0.1-0.3).
- **Positional control only one case.** Column swap was done at k=8 baseline; while it flipped P as expected (−0.99) — confirming the feature, not the column — we did not swap at every condition.

We openly acknowledge all of these in `proposal.md` §9 and `README.md`.

## 10. What can your experiment not establish?

**Question:** What can your experiment not establish?

**Expected answer:**

- **It cannot prove or disprove the claim in general.** We only tested one dataset family, one operationalization of simplicity, one model family. We can only make conditional claims.
- **It cannot show *causality* in the sense of a mechanism being the only one.** We showed a strong correlation between the complex feature's *isolated learnability* and the joint-task preference. The strongest plausible interpretation is optimization difficulty — but even then, at k=4 the complex feature is partially learnable yet P=+0.91 (competition/early latching). Our experiment establishes that optimization difficulty is a necessary part of the story.
- **It cannot establish that "simplicity bias" is intrinsic to SGD in the abstract.** It can establish what happens for this setup. The brief is clear: "a carefully designed experiment that fails to falsify the hypothesis can be more scientifically valuable than a poorly designed experiment that appears to contradict it."
- **It cannot rule out that a different architecture/initialization/loss would reveal a different preference.** We tested a small grid (lr ±3x, arch swap, budget ×5) — those are necessary but not exhaustive.
- **It cannot measure synergy.** Our non-combinable model means "both features useful" doesn't mean "combining them is better than either alone". Real-world shortcut learning often involves correlated features in more complicated ways.

## 11. What would you change if you had more compute or time?

**Question:** What would you change if you had more compute or time?

**Expected answer:** The following extensions would sharpen the conclusion:

1. **Mask forcing.** Train with `x_s` masked (or `x_c` masked) to be absolutely sure the complex feature is learnable in isolation, and also train *jointly* but enforce early masking or apply a schedule (warm-up: train on complex-only for a few epochs, then joint). This directly tests "does the easy feature's head start cause the complex feature to be lost?"
2. **Feature competition probing.** Use attribution / input-gradient norms, or probe representations. Also measure "when is the complex feature learned" — our checkpoints show preference is immediate at k=8, but not *where in the network*.
3. **More operationalizations.** Repeat k-sweep with (a) linear-separability-based complexity, (b) the Morwani et al. "low-dimensional input dependence" construction. Do we get the same boundary?
4. **Stronger strength-flip coverage.** Add strength flips at more k values (e.g. k=4) and also test (0.8,1.2) etc to see how steep the transition is.
5. **Robustness to seeds.** More seeds (n=10–20) would reduce noise and let us be precise about the null spread and the k=4 boundary.
6. **Optimization variants.** Test with full-batch GD (vs Adam), different init scales, or weight decay. This would test the "rich vs lazy" regime idea (Morwani et al.).
7. **Synergistic features.** Extend to a model where combining features *is* Bayes-better (e.g. OR/AND-like with independent channels) to see if the preference disappears.

**Practical note:** All of the above are "additions to a small experiment" — the current design already cleanly separates learnability from simplicity in the regimes where the answer is strongest (k=2 vs k=8). With more time the k=4 ambiguity (partly learnable, strongly dispreferred) would be the highest value target.

## Final caveat

The cleanest result: **at matched strength, preference for the simple feature tracks the complex feature's isolated learnability. When the complex feature is learnable, reliance follows predictive strength. When the complex feature is unlearnable, reliance is almost exclusively on the simple feature regardless of strength.** This qualifies the "simplicity bias" claim under our operationalization: the preference is conditional on the complex feature being hard to learn. We do not claim the claim is falsified; we claim it is conditionally supported and its mechanism is at least partly one of optimization difficulty, not of an intrinsic preference for simplicity.

*(All results are recorded in results/analysis_summary.md and figures in results/figures/; the full write-up is in proposal.md.)*
