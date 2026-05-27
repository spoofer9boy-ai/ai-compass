# Train-Val-Test Split

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 34 (ML Workflow Overview)  
**Estimated Time:** 35 minutes

## Why am I learning this?

You can build the most elegant model in the world, but if you evaluate it on the same data you trained it on, your metrics are fiction. The train-val-test split is the minimal hygiene that separates a demo from something you can ship. It is also the most common point of failure in early-stage ML teams: someone "tunes" hyperparameters by eyeballing test-set performance, leaks future information into training, or uses a random split on time-series data and reports a 99% accuracy that collapses in production.

This file exists so you never make those mistakes. You will learn the exact responsibilities of each split, the ratios that actually matter in practice, and the subtle ways data leakage creeps in. The goal is not to memorize a formula; it is to internalize a discipline: the test set is a contract with the future, and you are allowed to look at it exactly once.

## Where will I be using it?

- **Model development:** Training on the train set, selecting hyperparameters with the validation set, and reporting final numbers on the test set.
- **Kaggle competitions:** The "public leaderboard" is your validation set; the "private leaderboard" is your test set. Overfitting the public board is a rite of passage.
- **A/B testing infrastructure:** The test set becomes your holdout for measuring uplift before deploying a model.
- **Time-series forecasting:** Splitting by time instead of randomly, so you validate on the future and train on the past.
- **Medical or financial audits:** Regulators often require a locked test set that the modeling team never sees until final validation.

## Resources

- [scikit-learn Docs: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — The API you will call hundreds of times.
- [Google ML Crash Course: Splitting Data](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) — Practical guidance on ratios and leakage from Google's curriculum.
- [arXiv:1811.12808 — Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808) — A rigorous survey of resampling and evaluation protocols.
- [scikit-learn Docs: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — When a single validation fold is not enough.
- [fast.ai: Practical Deep Learning for Coders — Lesson 3 (Validation Sets)](https://course.fast.ai/) — Real-world perspective on why validation strategy determines whether your model works in production.

## Appendix

### Notation

- $\mathcal{D}_{\text{train}}$: Training set used to fit model parameters.
- $\mathcal{D}_{\text{val}}$: Validation set used to tune hyperparameters and select models.
- $\mathcal{D}_{\text{test}}$: Test set used for a single, final, unbiased estimate of generalization performance.

### Typical Ratios

| Dataset size | Train | Validation | Test |
|--------------|-------|------------|------|
| Small ($\lesssim 10^4$) | 60% | 20% | 20% |
| Medium ($10^4$–$10^6$) | 70% | 15% | 15% |
| Large ($\gtrsim 10^6$) | 80–98% | 1–10% | 1–10% |

With very large datasets, you can afford tiny validation and test sets because the law of large numbers gives you stable estimates anyway. With small datasets, prefer cross-validation to a single validation split.

### Common Pitfalls

- **Data leakage:** Preprocessing (scaling, imputation) must be fit on the training data only, then applied to validation and test. Fitting on the whole dataset before splitting leaks information.
- **Repeated test-set peeking:** Every time you look at test-set performance to make a modeling decision, you are effectively training on the test set. Lock it away.
- **Random splits on structured data:** If rows are not independent (e.g., multiple records per patient, user, or sensor), a random split can place correlated samples in train and test. Use group-based splits instead.
- **Temporal splits:** For time-dependent data, always split by time. A random split lets the model cheat by learning future patterns.

### Further Reading

- [Distill.pub: Feature-wise transformations](https://distill.pub/) — For deeper intuition on how data transformations interact with splits.
- [scikit-learn Docs: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) — When your data has group structure.
