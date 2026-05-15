# Expectation and Variance

**Phase:** PHASE-01-foundations  
**Prerequisites:** 24 (Random Variables)  
**Estimated Time:** 45 minutes

## Why am I learning this?

You will never hand-derive the expectation of a distribution in a production sprint. But you will spend an afternoon wondering why your model's loss oscillates wildly between batches, or why your A/B test needs three times more samples than your product manager expected. Expectation and variance are the two numbers that summarize every random variable you will ever meet — and most bugs in probabilistic systems come from confusing one for the other.

Expectation tells you where the distribution centers. Variance tells you how much it spreads. Together they let you answer questions like: "How many users do I need to detect a 2 % uplift?" or "Will my gradient estimator have finite variance?" If you skip this, you will be the engineer who runs an experiment for two weeks only to realize the effect size was smaller than the standard deviation.

## Where will I be using it?

- **A/B Testing:** Computing the required sample size from the variance of a Bernoulli conversion rate.
- **Deep Learning:** Analyzing gradient variance to diagnose why training is unstable; variance reduction techniques in reinforcement learning (e.g., control variates, baseline subtraction).
- **Model Evaluation:** Reporting mean ± std across k-fold cross-validation scores.
- **Optimization:** Understanding why mini-batch gradient descent has lower variance than single-sample SGD, and why that matters for convergence.
- **Bayesian Methods:** Propagating uncertainty through a pipeline by tracking both expected predictions and their predictive variance.
- **Data Quality:** Flagging anomalous sensors or features by thresholding on variance or z-score deviations from the mean.

## Resources

- [StatLect: Properties of the Expected Value](https://www.statlect.com/fundamentals-of-probability/expected-value-properties) — Rigorous summary of linearity, monotonicity, and transformation rules with proofs.
- [StatLect: Variance](https://www.statlect.com/fundamentals-of-probability/variance) — Clean definition via expected squared deviation, plus properties and solved exercises.
- [UvA Machine Learning Lecture Notes: Expectation and Variance](https://staff.fnwi.uva.nl/r.vandenboomgaard/MachineLearning/LectureNotes/ProbabilityStatistics/rvExpVar.html) — Concise bridge from probability theory to ML notation.
- [Wikipedia: Expected value](https://en.wikipedia.org/wiki/Expected_value) — Comprehensive reference with discrete, continuous, and measure-theoretic definitions.
- [PyTorch Docs: torch.var](https://pytorch.org/docs/stable/generated/torch.var.html) — The API you will actually use to compute variance over tensors.

## Appendix

### Notation

- $\mathbb{E}[X]$: Expected value (mean) of random variable $X$.
- $\mathrm{Var}(X)$: Variance of $X$, defined as $\mathbb{E}[(X - \mathbb{E}[X])^2]$.
- $\sigma_X = \sqrt{\mathrm{Var}(X)}$: Standard deviation of $X$.
- $\mathbb{E}[X \mid Y]$: Conditional expectation of $X$ given $Y$.

### Key Properties

**Linearity of expectation** (always holds, even when variables are dependent):

$$\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c$$

**Variance of a linear combination** (requires independence for the cross-term to vanish):

$$\mathrm{Var}(aX + bY) = a^2\,\mathrm{Var}(X) + b^2\,\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)$$

If $X$ and $Y$ are independent, $\mathrm{Cov}(X,Y) = 0$ and the formula simplifies.

**Alternative variance formula** (often easier for computation):

$$\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

### Common Pitfalls

- **Assuming independence:** $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ only when $X$ and $Y$ are independent (or at least uncorrelated). Violating this is a common source of bias in importance sampling and off-policy evaluation.
- **Confusing sample variance with population variance:** The unbiased estimator divides by $n - 1$, not $n$. PyTorch's `torch.var` defaults to unbiased (`unbiased=True`); NumPy's `np.var` defaults to population variance (`ddof=0`).
- **Ignoring variance in latency:** A pipeline with low average latency but high variance is worse for user experience than one with slightly higher average and tight bounds.

### Further Reading

- [Wikipedia: Variance](https://en.wikipedia.org/wiki/Variance) — Covers computational formulas, generalizations, and history.
- [3Blue1Brown: Binomial distributions](https://www.3blue1brown.com/lessons/binomial-distributions) — Visual intuition for how expectation and variance emerge from repeated trials.
