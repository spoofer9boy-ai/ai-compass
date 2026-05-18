# Maximum Likelihood Estimation

**Phase:** PHASE-01-foundations  
**Prerequisites:** 24 (Random Variables), 25 (Expectation and Variance), 26 (Common Distributions)  
**Estimated Time:** 55 minutes

## Why am I learning this?

You already know how to describe a distribution. Now you need to learn how to *choose* one. In production, you are rarely handed a clean generative model. You get a CSV of user session times, a histogram of image pixel values, or a stream of sensor readings, and someone asks: "What distribution does this follow?" Maximum Likelihood Estimation (MLE) is the standard answer. It gives you a principled way to pick the parameters of a distribution so that your observed data looks as probable as possible.

MLE is not just a statistics exercise. It is the optimization objective hiding inside many of the tools you already use. When scikit-learn fits a logistic regression, it is maximizing a likelihood. When a language model is trained, the cross-entropy loss is the negative log-likelihood of the next token. When you use `scipy.stats.norm.fit` to find the mean and standard deviation of a dataset, you are running MLE under the hood. Understanding MLE means understanding what these tools are actually optimizing, which makes debugging convergence failures and weird loss curves far easier.

## Where will I be using it?

- **Distribution Fitting:** Estimating the mean and variance of a Gaussian from data using `scipy.stats.norm.fit` or `torch.distributions.Normal`.
- **Logistic Regression:** The coefficients in scikit-learn's `LogisticRegression` are found by maximizing the binomial likelihood.
- **Language Models:** Cross-entropy loss is negative log-likelihood; training maximizes the likelihood of the observed token sequence.
- **Gaussian Mixture Models:** The EM algorithm alternates between computing responsibilities and maximizing the expected complete-data log-likelihood.
- **Survival Analysis:** Parametric models like the Weibull distribution use MLE to estimate time-to-event parameters from censored data.

## Resources

- [Wikipedia: Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) — Comprehensive overview of likelihood equations, properties, and common pitfalls.
- [CMU 36-705 Lecture 13: Point Estimation and MLE](http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf) — Rigorous derivation of MLE as a general-purpose estimator and comparison to Method of Moments.
- [CMU 36-705 Lecture 15: Asymptotic Theory for MLE](http://www.stat.cmu.edu/~larry/=stat705/Lecture15.pdf) — Consistency and asymptotic normality of the MLE under mild regularity conditions.
- [CMU 36-401 Lecture 6: MLE for Simple Linear Regression](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf) — Concrete walkthrough of deriving the MLE for Gaussian linear regression.
- [SciPy Docs: rv_continuous.fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html) — The API you will actually use to fit distributions via MLE.

## Appendix

### Notation

- $\mathcal{L}(\theta; \mathbf{y}) = \prod_{i=1}^{n} f(y_i; \theta)$: The likelihood function for i.i.d. data.
- $\ell(\theta; \mathbf{y}) = \ln \mathcal{L}(\theta; \mathbf{y})$: The log-likelihood, used for computational stability.
- $\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \ell(\theta; \mathbf{y})$: The maximum likelihood estimate.

### Common Pitfalls

- **Maximizing likelihood vs. log-likelihood:** They give the same optimum, but log-likelihood turns products into sums and avoids numerical underflow.
- **Ignoring support constraints:** If you fit an exponential distribution, the rate parameter must be positive. Unconstrained optimizers can return invalid values if you are not careful.
- **Local maxima:** For complex models (e.g., mixture models), the likelihood surface is non-convex. The EM algorithm is often used instead of direct gradient ascent.

### Further Reading

- [Deep Learning Book, Chapter 5: Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html) — Discusses MLE in the context of machine learning and its relationship to empirical risk minimization.
- [3Blue1Brown: Bayes' Theorem](https://www.3blue1brown.com/lessons/bayes-theorem) — While focused on Bayes, it builds the probabilistic intuition that makes MLE feel natural.
