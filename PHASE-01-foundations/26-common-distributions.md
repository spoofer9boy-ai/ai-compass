# Common Distributions

**Phase:** PHASE-01-foundations  
**Prerequisites:** 24 (Random Variables), 25 (Expectation and Variance)  
**Estimated Time:** 50 minutes

## Why am I learning this?

You will spend more time choosing and diagnosing distributions than you will deriving them. When a model's predictions look wrong, the first question is rarely "Is the architecture broken?" and usually "What distribution did the data actually come from?" If you assume a normal distribution for click-through rates, you will get nonsense confidence intervals. If you treat customer arrivals as uniform instead of Poisson, your server provisioning estimates will be off by an order of magnitude.

This file exists so you can look at a problem, name the distribution that governs it, and know immediately which properties you can rely on. You do not need to memorize every distribution in existence. You need to internalize the five or six that show up in almost every ML pipeline, understand when each one breaks, and know how to check your assumption with a few lines of code.

## Where will I be using it?

- **Regression residuals:** Assumed normal in ordinary least squares; checking this assumption is a standard diagnostic step.
- **Binary classification:** Logistic regression models the Bernoulli distribution; ensemble methods aggregate Bernoulli trials.
- **Count data:** Poisson and negative binomial distributions model event counts (page views, defects, arrivals).
- **Bayesian modeling:** Beta and Dirichlet distributions serve as conjugate priors for proportions and categorical parameters.
- **Initialization:** Xavier/He initialization draws weights from uniform or normal distributions with variances tuned to layer depth.
- **Generative models:** VAEs assume a normal latent space; diffusion models are built on Gaussian noise schedules.
- **A/B testing:** Binomial proportions power experiment calculators; normal approximations set confidence intervals.

## Resources

- [3Blue1Brown: Gaussian Integral](https://www.3blue1brown.com/lessons/gaussian-integral/) — Visual intuition for where the normal distribution comes from and why $\pi$ appears in it.
- [PyTorch Distributions](https://docs.pytorch.org/docs/stable/distributions.html) — The API you will use to sample, compute log-probabilities, and define loss terms.
- [scikit-learn: Map Data to a Normal Distribution](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html) — Practical examples of quantile transforms and power transforms to Gaussianize skewed data.
- [Wikipedia: Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution) — Canonical reference for the Gaussian: PDF, CDF, moments, and limiting behavior.
- [Wikipedia: Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution) — Formal derivation from the binomial limit and real-world event-count examples.

## Appendix

### Notation

- $X \sim \mathcal{N}(\mu, \sigma^2)$: Normal (Gaussian) random variable with mean $\mu$ and variance $\sigma^2$.
- $X \sim \text{Bernoulli}(p)$: Binary outcome with success probability $p$.
- $X \sim \text{Binomial}(n, p)$: Sum of $n$ independent Bernoulli$(p)$ trials.
- $X \sim \text{Poisson}(\lambda)$: Count of events in a fixed interval with rate $\lambda$.
- $X \sim \text{Uniform}(a, b)$: Equal probability over the interval $[a, b]$.
- $X \sim \text{Beta}(\alpha, \beta)$: Distribution over probabilities $[0, 1]$ with shape parameters $\alpha, \beta$.

### The Distributions in Detail

#### Normal (Gaussian)

The workhorse of ML. Arises naturally from the Central Limit Theorem: sums of independent random variables tend toward it.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

- **Mean:** $\mu$
- **Variance:** $\sigma^2$
- **When to use:** Continuous data with symmetric, bell-shaped variation; regression residuals; latent spaces in VAEs.
- **When it breaks:** Heavy tails (use Student's t or Laplace), bounded support (use Beta or log-normal), discrete counts (use Poisson or negative binomial).

#### Bernoulli

The simplest non-trivial distribution: a single coin flip.

$$P(X = k) = p^k (1 - p)^{1 - k}, \quad k \in \{0, 1\}$$

- **Mean:** $p$
- **Variance:** $p(1 - p)$
- **When to use:** Binary outcomes: click/no-click, fraud/no-fraud, pass/fail.

#### Binomial

Count of successes in $n$ independent Bernoulli trials.

$$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}$$

- **Mean:** $np$
- **Variance:** $np(1 - p)$
- **When to use:** Number of conversions in $n$ ad impressions, number of defective items in a batch.
- **Normal approximation:** Valid when $np \geq 5$ and $n(1 - p) \geq 5$.

#### Poisson

Models the number of events in a fixed interval when events occur independently at a constant rate.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- **Mean:** $\lambda$
- **Variance:** $\lambda$
- **When to use:** Server requests per second, customer arrivals per hour, radioactive decays per minute.
- **Key property:** If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ are independent, then $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$.
- **When it breaks:** Overdispersion (variance > mean). Use negative binomial instead.

#### Uniform

Every outcome in the interval is equally likely.

$$f(x) = \frac{1}{b - a}, \quad x \in [a, b]$$

- **Mean:** $\frac{a + b}{2}$
- **Variance:** $\frac{(b - a)^2}{12}$
- **When to use:** Random initialization bounds, sampling from a continuous range when no prior information exists, dropout masks.

#### Beta

A distribution over probabilities. Defined on $[0, 1]$ and parameterized by two shape parameters.

$$f(x) = \frac{x^{\alpha - 1}(1 - x)^{\beta - 1}}{B(\alpha, \beta)}$$

- **Mean:** $\frac{\alpha}{\alpha + \beta}$
- **Variance:** $\frac{\alpha\beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$
- **When to use:** Bayesian priors for proportions, Thompson sampling in bandit algorithms, modeling uncertainty in click-through rates.

### Common Pitfalls

- **Assuming normality for everything.** Skewed financial returns, bounded user ratings, and sparse click data are not Gaussian. Plot a histogram before you model.
- **Ignoring overdispersion.** If your count data has variance much larger than its mean, Poisson is the wrong model. Negative binomial or zero-inflated models are the usual fixes.
- **Using the normal approximation too early.** For small $n$ or extreme $p$, the binomial is asymmetric and the normal approximation will mislead you.
- **Forgetting that distributions have domains.** The normal distribution spans $(-\infty, \infty)$. If your variable is strictly positive (heights, prices), a log-normal or Gamma distribution may be more appropriate.

### Further Reading

- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) — Comprehensive reference for PDFs, CDFs, and sampling methods in Python.
- [Wikipedia: Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution) — Deep dive into the distribution of distributions.
- [Wikipedia: Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution) — Formal properties, history, and relationship to the normal distribution.
