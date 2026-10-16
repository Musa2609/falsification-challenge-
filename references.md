# References

All bibliographic details below were verified against the publisher/arXiv pages
(checked 2026-10-16). Claims attributed to each paper were verified against the
abstract and, where quoted, the paper text.

---

## Brief-recommended literature

### 1. The Pitfalls of Simplicity Bias in Neural Networks (primary)
- **Authors:** Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain,
  Praneeth Netrapalli
- **Venue:** Advances in Neural Information Processing Systems 33 (NeurIPS 2020),
  Vancouver.
- **arXiv:** 2006.07710 (v1 13 Jun 2020; v2 28 Oct 2020). DOI:
  10.48550/arXiv.2006.07710
- **Links:** https://arxiv.org/abs/2006.07710 ·
  https://proceedings.neurips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html
- **Central claims (as stated by the authors):**
  - It introduces piecewise-linear and image-based datasets that "incorporate a
    precise notion of simplicity" and "comprise multiple predictive features with
    varying levels of simplicity."
  - Observation (i): SB of SGD and variants "can be extreme: neural networks can
    exclusively rely on the simplest feature and remain invariant to all
    predictive complex features."
  - Observation (iii): "SB can also hurt generalization on the same data
    distribution, as SB persists even when the simplest feature has less
    predictive power than the more complex features." ← the observation our
    strength control targets.
  - Observation (iv): ensembles and adversarial training "can fail in mitigating
    SB and its pitfalls."
  - The authors explicitly note "the precise notion of simplicity remains vague,"
    and prove a 1-hidden-layer/hinge-loss result (Theorem 1) on one of their
    synthetic distributions.
- **How used:** This is the claim under test. Our synthetic task mirrors its
  structure (multiple features of varying simplicity and predictive strength,
  combined so that each alone suffices). Our strength control is a direct test of
  observation (iii); our isolated-learnability control is the extra factor the
  original observation does not isolate.

### 2. Simplicity Bias in 1-Hidden Layer Neural Networks (supporting)
- **Authors:** Depen Morwani, Jatin Batra, Prateek Jain, Praneeth Netrapalli
- **Venue:** Advances in Neural Information Processing Systems 36 (NeurIPS 2023).
- **arXiv:** 2302.00457 (1 Feb 2023).
- **Links:** https://arxiv.org/abs/2302.00457 ·
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html
- **Central claims:**
  - Defines a particular simplicity bias for 1-hidden-layer networks as
    **low-dimensional input dependence** ("LD-SB"): the model is essentially a
    function of a low-dimensional projection of the inputs.
  - Theoretically shows that for the *independent features model* (IFM) with
    linearly separable data, the network "primarily depends on only the linearly
    separable (1-dimensional) subspace even in the presence of an arbitrarily
    large number of other, more complex features."
  - Empirically verifies LD-SB on real datasets (Imagenette, Waterbirds-Landbirds).
  - Notes that training trajectory depends on the scale of initialisation (rich
    vs lazy regime).
- **How used:** Motivates the *independent features* construction and the
  observation that SB is, at least partly, an optimisation/initialisation
  statement — supporting our decision to measure the complex feature's isolated
  learnability.

### 3. On the Spectral Bias of Neural Networks (optional background)
- **Authors:** Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler,
  Min Lin, Fred Hamprecht, Yoshua Bengio, Aaron Courville
- **Venue:** Proceedings of the 36th International Conference on Machine Learning
  (ICML 2019), PMLR 97:5301–5310. *(The brief lists this as ICML 2019; that is
  correct — verified at proceedings.mlr.press/v97/rahaman19a.html.)*
- **arXiv:** 1806.08734
- **Links:** https://proceedings.mlr.press/v97/rahaman19a.html ·
  https://arxiv.org/abs/1806.08734
- **Central claims:** Neural networks fit low-frequency components of a target
  function before high-frequency ones (spectral bias); the geometry of the data
  manifold affects the behaviour.
- **How used:** Provides the "low-frequency / high-frequency" sense of simplicity
  that our complexity knob (number of decision boundaries = 2k) broadly follows,
  and the theoretical backdrop for the learning-curve analysis (complex, higher-
  frequency structure learned later, if at all).

---

## Software cited

- NumPy, SciPy, scikit-learn, PyTorch (CPU), Matplotlib, pytest. Exact versions
  are recorded in `requirements.txt` and in `results/run_manifest.json` for the
  run that produced the results.

## Note on citation honesty

- No source is cited that was not retrieved and checked. The three references
  above are the extent of the literature used; the brief explicitly advises
  against an exhaustive review when a few directly relevant papers suffice.
- Where a claim is a *paraphrase* of an abstract, it is marked as "as stated by
  the authors." No quotation marks are used around paraphrases.
