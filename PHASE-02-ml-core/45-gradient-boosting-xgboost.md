# Gradient Boosting and XGBoost

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 43, 44  
**Estimated Time:** 60 minutes

## Why am I learning this?

You have already trained single decision trees and random forests. In many real-world competitions and production pipelines, they are not the final model. Gradient boosting is. Kaggle leaderboards, production fraud-detection systems, and ad-click predictors at major tech companies are dominated by gradient boosted trees because they squeeze more predictive power out of tabular data than almost anything else.

The core idea is simple but powerful: instead of training one big model, train a sequence of small models—usually shallow trees—where each new model tries to correct the errors of the combined ensemble so far. The "gradient" part means you treat the residual (the error) as a target and fit a new model to the negative gradient of your loss function. After enough rounds, the weighted sum of these weak learners becomes a strong predictor. It is an optimization algorithm disguised as an ensemble method.

XGBoost is the library that made this idea practical at scale. It adds regularization (penalties on tree complexity), approximate split finding, column and row subsampling, and cache-aware block storage. The result is a tool that trains faster, overfits less, and wins competitions. If you work with structured data, you will use XGBoost or a close descendant (LightGBM, CatBoost) regularly.

## Where will I be using it?

- **Fraud detection:** Classifying transactions in real-time with highly imbalanced tabular data.
- **Click-through rate prediction:** Ad-tech stacks at major platforms use boosted trees as baseline and sometimes final models.
- **Credit scoring:** Regulatory-friendly models where feature importance and monotonicity constraints matter.
- **Recommendation systems:** Ranking candidate items when user and item features are tabular.
- **Feature importance analysis:** Even when a neural net is the final model, XGBoost is often used to sanity-check which features actually matter.

## Resources

- [XGBoost: A Scalable Tree Boosting System (arXiv:1603.02754)](https://arxiv.org/abs/1603.02754) — The original paper introducing the system design, regularization, and split-finding algorithms.
- [XGBoost Documentation: Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) — Official walkthrough of the gradient boosting objective and tree structure.
- [scikit-learn: Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) — Practical API reference and comparison with other ensembles.
- [How to Explain Gradient Boosting (explained.ai)](https://explained.ai/gradient-boosting/) — Intuitive, visual derivation of the algorithm without skipping the math.
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) — The canonical reference for tuning `eta`, `max_depth`, `subsample`, `colsample_bytree`, and regularization terms.

## Appendix

### Notation

- $F_M(x) = \sum_{m=1}^M \eta \cdot h_m(x)$: The final ensemble after $M$ rounds with learning rate $\eta$.
- $r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$: The negative gradient (pseudo-residual) for sample $i$ at round $m$.
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$: XGBoost regularization term, penalizing leaf weights $w_j$ and number of leaves $T$.

### Common Pitfalls

- Setting `max_depth` too high. Gradient boosting works because each tree is weak; a depth of 3–6 is usually enough.
- Forgetting to tune the learning rate (`eta`) alongside `n_estimators`. A lower `eta` needs more rounds.
- Ignoring `scale_pos_weight` on imbalanced datasets. The default assumes roughly balanced classes.
- Treating XGBoost as a black box without checking feature importance or SHAP values. Overfitting can hide in unexpected features.

### Further Reading

- [LightGBM Documentation](https://lightgbm.readthedocs.io/) — Microsoft's alternative using leaf-wise tree growth and histogram-based splits.
- [CatBoost Documentation](https://catboost.ai/) — Yandex's library with native categorical feature handling and ordered boosting.
