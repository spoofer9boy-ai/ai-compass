# Conditional Probability

**Phase:** PHASE-01-foundations  
**Prerequisites:** 21 (Probability Axioms)  
**Estimated Time:** 40 minutes

## Why am I learning this?

You will never write `P(A|B)` in production code. But you will spend hours debugging why a spam filter flags legitimate emails, why a medical diagnostic model gives bizarre confidence scores on edge cases, or why an A/B test result contradicts common sense. In every one of those situations, the root cause is a misunderstanding of how probabilities update when new information arrives.

Conditional probability is the engine behind every Bayesian model, every recommendation system that adjusts to user behavior, and every fraud-detection pipeline that re-scores transactions as new signals appear. It is also the concept that separates engineers who treat ML as a black-box API from engineers who can reason about why a model behaves the way it does.

The notation looks innocent—just a vertical bar—but it encodes a shift in perspective from "what is the world like?" to "what is the world like *now that I know this*?". Mastering that shift is non-negotiable if you want to build systems that learn from data rather than merely memorize it.

## Where will I be using it?

- **Bayesian Networks:** Updating belief states as evidence streams in; the entire field of probabilistic graphical models rests on conditional probability.
- **Natural Language Processing:** Language models compute `P(next_token | previous_tokens)`; autoregressive decoding is conditional probability at scale.
- **Recommender Systems:** Computing `P(click | user_history, item_features)` to rank recommendations.
- **Fraud Detection:** Re-scoring transaction risk as new events occur: `P(fraud | location_mismatch AND velocity_alert)`.
- **A/B Testing:** Understanding that `P(conversion | variant_A)` is not the same as `P(variant_A | conversion)`—confusing the two is the canonical base-rate fallacy.
- **Medical AI:** Interpreting diagnostic test results where disease prevalence (the prior) dramatically changes the posterior probability.

## Resources

- [Khan Academy: Conditional Probability](https://en.khanacademy.org/math/statistics-probability/probability-library/conditional-probability-independence/v/conditional-probability2) — Visual intuition with concrete examples.
- [Math is Fun: Conditional Probability](https://www.mathsisfun.com/data/probability-events-conditional.html) — Gentle, example-driven introduction with tree diagrams.
- [Wikipedia: Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability) — Formal definition, properties, and measure-theoretic foundations.
- [Deep Learning Book, Chapter 3](https://www.deeplearningbook.org/contents/prob.html) — Goodfellow et al.; rigorous treatment of probability and information theory for ML practitioners.
- [arXiv: On the Computability of Conditional Probability](https://arxiv.org/abs/1005.3014v3) — Ackerman, Freer, and Roy; theoretical grounding on why conditional probability is subtle in continuous and computable settings.

## Appendix

### Notation

- $P(A)$: Probability of event $A$ occurring.
- $P(A \mid B)$: Probability of $A$ occurring *given* that $B$ has occurred.
- $P(A \cap B)$ or $P(A, B)$: Joint probability of both $A$ and $B$ occurring.

### Definition

For events $A$ and $B$ with $P(B) > 0$:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

Equivalently, the **product rule** (or chain rule for two events):

$$
P(A \cap B) = P(A \mid B) \, P(B) = P(B \mid A) \, P(A)
$$

### Independence

Two events $A$ and $B$ are **independent** if knowing one does not change the probability of the other:

$$
P(A \mid B) = P(A) \quad \text{or equivalently} \quad P(A \cap B) = P(A) \, P(B)
$$

Do not confuse independence with mutual exclusivity. Mutually exclusive events ($P(A \cap B) = 0$) are *highly* dependent: if one happens, the other definitely does not.

### Common Pitfalls

- **Base-rate neglect:** In medical testing, a positive result does not mean $P(\text{disease} \mid \text{positive}) \approx 1$. You must account for disease prevalence $P(\text{disease})$ and test specificity.
- **Confusing $P(A \mid B)$ with $P(B \mid A)$:** The probability of symptoms given a disease is not the probability of disease given symptoms. Bayes' theorem (the next subject) is the formal fix.
- **Assuming independence without justification:** Naive Bayes classifiers assume feature independence; this is often false, but the model can still work surprisingly well if the decision boundary is simple enough.

### Further Reading

- [3Blue1Brown: Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) — Visual proof and intuition for the relationship between conditional probability and Bayes' theorem.
- [StatQuest: Conditional Probability](https://www.youtube.com/watch?v=_IgyaD7vOOA) — Josh Starmer's concise explanation with real-world analogies.
