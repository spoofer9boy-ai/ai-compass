# Random Forest

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 42 (Decision Trees), 43 (Ensemble Methods)  
**Estimated Time:** 50 minutes

## Why am I learning this?

A single decision tree is fast, interpretable, and easy to explain to a product manager. It is also unstable: a small change in training data can flip the root split and produce a completely different tree. In production, that instability translates to unpredictable validation accuracy and models that break silently when the data distribution shifts.

Random Forest fixes this by building many trees on random subsets of the data and averaging their predictions. The idea is old—Leo Breiman formalized it in 2001—but it is still one of the most reliable baselines in applied machine learning. If you are handed a tabular dataset and asked to produce a working model in an afternoon, a tuned Random Forest is often the first thing you should try. It requires little preprocessing, handles mixed feature types, and gives you feature importance for free. It will not beat a carefully engineered gradient boosting model on every benchmark, but it will rarely embarrass you, and it trains fast enough that you can iterate quickly.

## Where will I be using it?

- **Tabular data baselines:** Kaggle competitions, internal dashboards, and proof-of-concept models where you need a strong default quickly.
- **Feature selection:** The built-in importance scores (mean decrease in impurity or permutation importance) tell you which columns actually matter before you move to a more expensive model.
- **Anomaly detection:** Isolation Forest, a variant of Random Forest, is the standard approach for detecting outliers in high-dimensional tabular data.
- **Production pipelines:** scikit-learn's `RandomForestClassifier` and `RandomForestRegressor` are mature, well-documented, and integrate cleanly with existing ML pipelines.
- **Interpretability workflows:** While less transparent than a single tree, partial dependence plots and SHAP values computed on a Random Forest are standard tools for explaining model behavior to stakeholders.

## Resources

- [scikit-learn: Ensemble Methods — Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) — The API you will actually use, with clear explanations of hyperparameters.
- [Breiman (2001): Random Forests](https://doi.org/10.1023/A:1010933404324) — The original paper. Dense but definitive if you want to understand why bagging + random feature subsets works.
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — A visual, intuition-first walkthrough of how the algorithm builds and aggregates trees.
- [scikit-learn: Feature Importances with Forests of Trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) — Practical guide to extracting and interpreting feature importance scores.
- [arXiv: Decision-Path Patterns as Tree Reliability Signals](https://arxiv.org/abs/2605.20716) — Recent work on adaptive weighting in Random Forest classification, showing the algorithm is still an active research target.

## Appendix

### How it works

1. **Bootstrap sampling (bagging):** From a dataset with $n$ samples, draw $n$ samples with replacement to create a new training set. Repeat this $B$ times to get $B$ datasets.
2. **Tree training with random subspaces:** For each bootstrap dataset, train a decision tree. At every split, consider only a random subset of $m$ features (typically $m = \sqrt{p}$ for classification, $m = p/3$ for regression, where $p$ is the total number of features).
3. **Aggregation:** For classification, take a majority vote across all trees. For regression, average the predictions.

### Key hyperparameters

- `n_estimators`: Number of trees. More is usually better, with diminishing returns. Start at 100–500.
- `max_features`: Number of features to consider at each split. `sqrt` for classification, `log2` or a fraction for regression.
- `max_depth`: Maximum depth of each tree. Deeper trees reduce bias but increase variance. Random Forest is somewhat robust to overfitting from depth, but very deep trees slow training.
- `min_samples_leaf`: Minimum samples required at a leaf node. Increasing this smooths the model and reduces overfitting.

### Common Pitfalls

- **Trusting default importances:** scikit-learn's default feature importance is based on mean decrease in impurity, which is biased toward high-cardinality features. Use permutation importance for a more reliable estimate.
- **Ignoring inference cost:** A forest with 500 trees is 500× slower at prediction than a single tree. If latency matters, consider trimming the forest or switching to a linear model.
- **Tuning every knob:** Random Forest is robust. Grid-searching every parameter is usually a waste of time; focus on `n_estimators`, `max_features`, and `max_depth`.

### Further Reading

- [scikit-learn: Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html) — How to compute unbiased feature importance.
- [Distill.pub: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — Not about forests directly, but a good reminder that even robust models need careful validation.
