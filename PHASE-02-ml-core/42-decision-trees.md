# Decision Trees

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 34  
**Estimated Time:** 55 minutes

## Why am I learning this?

You will rarely ship a single decision tree in production, but you will spend a lot of time explaining model behavior to stakeholders who do not trust black boxes. Decision trees are the only widely used ML model that a non-technical product manager can read and question directly. Understanding how they split, grow, and overfit is the foundation for every ensemble method you will actually deploy—Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost all start here.

Beyond interpretability, trees force you to confront the core tension in supervised learning: the bias-variance tradeoff. A shallow tree is stable but underfits; a deep tree memorizes the training set and collapses on new data. Learning to control this with hyperparameters like `max_depth`, `min_samples_leaf`, and pruning is a transferable skill that applies to nearly every model you will tune.

## Where will I be using it?

- **Fraud Detection:** Isolation Forests and rule-based risk engines often begin with a single interpretable tree to establish baseline logic before stacking ensembles.
- **Healthcare Diagnostics:** Clinicians and regulators frequently require explainable models. A decision tree can satisfy compliance while providing a starting point for more complex pipelines.
- **Customer Segmentation:** Marketing teams use trees to generate human-readable segments (e.g., "users who signed up in the last 30 days and spent over $50") that drive campaign targeting.
- **Feature Importance Prototyping:** Before training a costly ensemble, a quick decision tree run reveals which features carry signal and which are noise.
- **Baseline Modeling:** In Kaggle competitions and internal benchmarks, a tuned decision tree is the first sanity-check model before moving to gradient boosting.

## Resources

- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html) — Official documentation covering classification, regression, and all splitting criteria.
- [Google Developers: Decision Forests Course](https://developers.google.com/machine-learning/decision-forests) — Comprehensive introduction to how decision trees and forests work, with YDF code examples.
- [Wikipedia: Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning) — Solid reference on algorithm history, splitting metrics, and variants like CART, ID3, and C4.5.
- [CMU 10-601: Decision Trees Lecture](https://www.cs.cmu.edu/~bhiksha/courses/10-601/decisiontrees/) — University lecture notes with step-by-step construction and information gain derivation.
- [arXiv:2207.08815 — Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) — Recent empirical study reinforcing why trees remain dominant for structured data.

## Appendix

### Notation

- $\mathbf{x} \in \mathbb{R}^{d}$: Input feature vector with $d$ dimensions.
- $y \in \{0, 1, \dots, K-1\}$: Target class label for classification.
- $G(t)$: Gini impurity at node $t$.
- $H(t)$: Entropy (information) at node $t$.

### Common Splitting Criteria

| Criterion | Formula | Best For |
|---|---|---|
| Gini Impurity | $G = 1 - \sum_{k} p_k^2$ | Faster computation, CART default |
| Entropy / Information Gain | $H = -\sum_{k} p_k \log p_k$ | Theoretically grounded, ID3/C4.5 |
| Mean Squared Error | $\text{MSE} = \frac{1}{N} \sum (y_i - \bar{y})^2$ | Regression trees |

### Common Pitfalls

- **Overfitting by default:** An unpruned tree will memorize noise. Always set `max_depth`, `min_samples_split`, or `min_samples_leaf`.
- **Biased splits on dominant classes:** Imbalanced targets lead to trivial trees that always predict the majority class. Use `class_weight` or resample.
- **Extrapolation failure:** Trees predict piecewise constant values and cannot extrapolate beyond the training range—critical for time-series or continuous regression tasks.
- **Unstable structure:** Small data perturbations can yield completely different trees. This is why ensembles (bagging, boosting) exist.

### Further Reading

- [Breiman et al., *Classification and Regression Trees* (1984)](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman) — The original CART monograph.
- [Quinlan, *Induction of Decision Trees* (1986)](https://doi.org/10.1023/A:1022643204877) — Foundational paper on ID3 and information gain.
- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — While focused on neural nets, the interpretability mindset applies directly to tree inspection.
