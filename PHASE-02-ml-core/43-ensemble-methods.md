# Ensemble Methods

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 42 (Decision Trees)  
**Estimated Time:** 45 minutes

## Why am I learning this?

A single decision tree is easy to interpret, but it is also fragile: a small change in training data can produce a completely different tree, and its depth-limited leaves often underfit or overfit. In production, you will rarely ship a lone tree. You will ship an ensemble—a collection of models whose combined prediction is more stable and accurate than any individual member.

Ensemble methods are the default choice for structured-data problems in industry. Kaggle competitions are dominated by gradient-boosted ensembles. Fraud-detection systems at banks stack a forest of trees with logistic regression. Recommendation systems blend collaborative and content-based models. Even deep-learning pipelines sometimes fall back to tree ensembles when the dataset is tabular and the sample size is modest. Understanding how these combinations work—and where they fail—saves you from training one giant model when a committee of small ones would do better, faster, and cheaper.

The core idea is simple: if you have a set of models that make uncorrelated errors, averaging their predictions cancels out the mistakes and amplifies the correct signal. The art is in how you create that diversity and how you aggregate the results.

## Where will I be using it?

- **Tabular ML pipelines:** Fraud detection, credit scoring, churn prediction, and customer-lifetime-value models at banks and fintechs almost always use XGBoost, LightGBM, or CatBoost ensembles.
- **Search ranking:** Google and Bing use gradient-boosted decision trees (and ensembles of them) as part of their learning-to-rank stacks.
- **Recommendation systems:** Stacking collaborative filtering with content-based models to cover cold-start and popularity-bias gaps.
- **Model competitions:** Winning solutions on Kaggle typically blend 5–50 models with stacking or weighted averaging.
- **Uncertainty estimation:** Bagging provides a cheap way to estimate prediction variance without Bayesian inference.
- **Anomaly detection:** Isolation Forests (an ensemble of randomized trees) are a standard baseline for outlier detection.

## Resources

- [scikit-learn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) — Official documentation covering bagging, boosting, voting, and stacking with code examples.
- [Understanding Random Forests: From Theory to Practice](https://arxiv.org/abs/1407.7502) — Deep theoretical and empirical treatment of bagging and random forests.
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — The paper that introduced the algorithm behind most production gradient-boosting pipelines.
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) — NeurIPS paper on histogram-based gradient boosting used in large-scale industry systems.
- [Popular Ensemble Methods: An Empirical Study](https://arxiv.org/abs/1106.0257) — Comparative analysis of bagging, boosting, and stacking across datasets.

## Appendix

### Notation

- $h_b(x)$: Prediction of base learner $b$ on input $x$.
- $H(x)$: Final ensemble prediction, often $\frac{1}{B}\sum_{b=1}^{B} h_b(x)$ for regression or majority vote for classification.
- $B$: Number of base learners in the ensemble.

### The three families

1. **Averaging methods (Bagging):** Train multiple models in parallel on random subsets of data, then average. Random Forest is the canonical example.
2. **Boosting methods:** Train models sequentially, with each new model focusing on the errors of the previous one. AdaBoost, Gradient Boosting, XGBoost, LightGBM.
3. **Stacking:** Train a meta-learner to combine the predictions of heterogeneous base models.

### Why diversity matters

If all base learners make the same error, averaging does not help. Diversity can come from:
- Different training subsets (bagging)
- Different feature subsets (random forests)
- Different learning algorithms (stacking)
- Sequential reweighting of hard examples (boosting)

### Common pitfalls

- **Overfitting with boosting:** Adding too many trees can overfit; use early stopping and validation curves.
- **Ignoring calibration:** Averaging raw probabilities from poorly calibrated models can produce misleading confidence scores.
- **Data leakage in stacking:** The meta-learner must be trained on out-of-fold predictions, not the same data the base learners saw.
- **Computational cost:** Ensembles are harder to interpret and slower at inference than single models. Tree-based ensembles can be large in memory.

### Further Reading

- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — Not ensemble-specific, but useful for understanding model behavior when you move beyond single estimators.
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516) — If your tabular data has many categorical variables.
