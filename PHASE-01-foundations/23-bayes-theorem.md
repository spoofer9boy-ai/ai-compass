# Bayes Theorem

**Phase:** PHASE-01-foundations  
**Prerequisites:** 22 (Conditional Probability)  
**Estimated Time:** 45 minutes

## Why am I learning this?

You already know how to update a belief when you see new evidence—at least informally. Bayes theorem is the formal rule that tells you *exactly how much* to update. In production ML, you will use it directly (Naive Bayes classifiers) and indirectly (Bayesian optimization, posterior inference in probabilistic models, and even A/B test analysis). The theorem is also the source of the most common probability mistake in engineering: treating $P(A \mid B)$ as if it were $P(B \mid A)$. If you have ever seen a model output a "probability" and wondered whether it was a likelihood or a posterior, this file is the fix.

Bayes theorem is not magic. It is a rearrangement of the definition of conditional probability. Its power comes from the fact that the world usually gives us likelihoods—$P(\text{evidence} \mid \text{hypothesis})$—while we actually want posteriors—$P(\text{hypothesis} \mid \text{evidence})$. The theorem bridges that gap.

## Where will I be using it?

- **Spam filtering:** Updating $P(\text{spam} \mid \text{words})$ as new emails arrive.
- **Medical testing:** Computing the real probability of a disease given a positive test, which is often much lower than the test's accuracy.
- **Naive Bayes classifiers:** Fast baseline text classifiers in scikit-learn.
- **Bayesian optimization:** Updating a surrogate model's belief about where the optimum lies.
- **A/B testing:** Combining prior experiment data with new observations to decide which variant wins.

## Resources

- [3Blue1Brown: Bayes' theorem](https://www.3blue1brown.com/lessons/bayes-theorem/) — Visual intuition for how evidence reshapes belief.
- [Wikipedia: Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) — Formal statement, derivations, and history.
- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) — The production API that applies Bayes theorem at scale.
- [Tom Rocks Maths: Ghosts, Spam Emails and Bayes' Theorem](https://tomrocksmaths.com/2021/08/24/ghosts-spam-emails-and-bayes-theorem/) — Concrete real-world examples including spam filtering.
- [Statistics Reference Guide: Real World Examples](https://statistics.reference.guide/bayes-theorem/real-world-examples) — Industry applications across healthcare, finance, and engineering.

## Appendix

### Notation

- $P(H)$: Prior probability of hypothesis $H$.
- $P(E \mid H)$: Likelihood of evidence $E$ given $H$.
- $P(H \mid E)$: Posterior probability of $H$ after observing $E$.
- $P(E)$: Total probability of the evidence (marginal likelihood).

### Theorem

$$P(H \mid E) = \frac{P(E \mid H) \, P(H)}{P(E)}$$

Where $P(E) = \sum_i P(E \mid H_i) P(H_i)$ for a partition of hypotheses.

### Common Pitfalls

- **Confusing likelihood and posterior.** $P(E \mid H)$ is not $P(H \mid E)$.
- **Ignoring the prior.** A strong prior can dominate weak evidence; neglecting it leads to overreaction.
- **Base rate fallacy.** In medical testing, people intuitively use test sensitivity ($P(\text{positive} \mid \text{disease})$) instead of the actual probability of disease given a positive test.

### Further Reading

- [arXiv: Bayesian inference: More than Bayes's theorem](https://arxiv.org/abs/2406.18905) — A deeper look at how Bayesian inference uses all of probability theory, not just the theorem itself.
