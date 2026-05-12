# Probability Axioms

**Phase:** PHASE-01-foundations  
**Prerequisites:** []  
**Estimated Time:** 35 minutes

## Why am I learning this?

You will never write `P(A) = 0.7` in production and wonder if the math is legal. But you will spend hours debugging why a variational autoencoder's ELBO drifts negative, or why a softmax output sums to 1.0000001 instead of 1, or why a Monte Carlo estimator explodes. Those bugs trace back to the same three rules: non-negativity, normalization, and countable additivity. This file exists so that when you see `log p(x)` in a loss function, you know exactly what assumptions you are standing on—and where they break.

Probability axioms are the contract that makes all of machine learning possible. Every loss function that contains an expectation, every posterior that you sample from, and every confidence score that a classifier emits is valid only because someone proved it follows from these three rules. Understanding them lets you spot when a model's outputs are not actually probabilities (temperature-scaled logits, uncalibrated sigmoids) and why fixing that matters for decision-making under uncertainty.

## Where will I be using it?

- **Softmax outputs:** The final layer of any classifier is a probability distribution over classes, built by enforcing non-negativity (exp) and normalization (divide by sum).
- **Loss functions:** Cross-entropy loss assumes the model outputs a valid probability distribution. If they do not sum to 1, the loss is not the true cross-entropy.
- **Generative models:** VAEs, diffusion models, and normalizing flows all manipulate probability densities. Sampling and density evaluation require valid measures.
- **Reinforcement learning:** Policy gradients estimate expectations over trajectories. The expectation is well-defined only because the trajectory distribution satisfies the axioms.
- **Bayesian optimization:** Acquisition functions like Expected Improvement rely on posterior distributions that must be normalized probability measures.

## Resources

- [Probability Axioms — Wikipedia](https://en.wikipedia.org/wiki/Probability_axioms) — Clean statement of Kolmogorov's three axioms with motivation and history.
- [What is the significance of the Kolmogorov axioms? — David Aldous, UC Berkeley](https://www.stat.berkeley.edu/~aldous/Real_World/kolmogorov.html) — Short essay on why measure-theoretic foundations matter in practice.
- [Kolmogorov axioms of probability — The Book of Statistical Proofs](https://statproofbook.github.io/D/prob-ax.html) — Formal definitions with proof templates for derived rules.
- [Probability distributions — PyTorch Docs](https://pytorch.org/docs/stable/distributions.html) — The API you will actually use to construct and sample from distributions.
- [Probability in Practice: A Hands-On Journey with Python](https://snowch.github.io/learn_probability/index.html) — Practical probability implementations with real-world datasets.

## Appendix

### Notation

- $\Omega$: The sample space (set of all possible outcomes).
- $\mathcal{F}$: A $\sigma$-algebra (collection of events we can assign probability to).
- $P: \mathcal{F} \to [0, 1]$: A probability measure.
- $A \in \mathcal{F}$: An event (a subset of $\Omega$).

### The Three Axioms

1. **Non-negativity:** For any event $A$, $P(A) \geq 0$.
2. **Normalization:** $P(\Omega) = 1$.
3. **Countable additivity:** For any countable sequence of disjoint events $A_1, A_2, \ldots$,
   $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

### Derived Rules (used constantly in ML)

- **Complement:** $P(A^c) = 1 - P(A)$
- **Union bound:** $P(A \cup B) \leq P(A) + P(B)$
- **Inclusion-exclusion:** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### Common Pitfalls

- **Confusing probability with probability density:** Densities can exceed 1; probabilities cannot. A PDF value of 2.5 at a point is perfectly valid.
- **Assuming finite additivity is enough:** In continuous spaces, you need countable additivity to handle limits and expectations properly.
- **Forgetting the sample space:** $P(\Omega) = 1$ is what makes softmax a valid distribution. If you forget the denominator, you have unnormalized scores, not probabilities.

### Further Reading

- [SticiGui Probability: Axioms and Fundaments — UC Berkeley](https://www.stat.berkeley.edu/~stark/SticiGui/Text/probabilityAxioms.htm) — Interactive textbook chapter with exercises.
- [d2l-pytorch: Probability and Statistics](https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch04_The_Preliminaries_A_Crashcourse/Probability_and_Statistics.ipynb) — Notebook covering probability fundamentals with PyTorch code.
