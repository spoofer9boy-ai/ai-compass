# Classification Metrics

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 40 (Train-Val-Test Split)  
**Estimated Time:** 50 minutes

## Why am I learning this?

You have split your data, trained a classifier, and now you have a model that outputs predictions. The next question is unavoidable: "How good is it?" In a professional setting, this question is never answered with a single number. A product manager will ask about precision because false positives cost money. A medical team will ask about recall because missing a positive case can cost lives. Your job as an engineer is to know which metric answers which question, and to never let a single score hide the full story.

Accuracy is the metric everyone reaches for first, and it is often the wrong one. If you are detecting fraud in a dataset where 99% of transactions are legitimate, a model that predicts "legitimate" every single time will score 99% accuracy while being completely useless. This file exists so you know when accuracy is lying to you, and what to use instead.

Understanding classification metrics also protects you in code review and stakeholder meetings. Someone will ask why you optimized for F1 instead of accuracy. Someone will ask why the ROC AUC is high but the PR curve looks bad. The answers come down to the definitions in this file.

## Where will I be using it?

- **Model evaluation pipelines:** Choosing the right scoring function in scikit-learn's `cross_val_score` or GridSearchCV.
- **Imbalanced datasets:** Fraud detection, spam filtering, medical diagnosis, and rare-event prediction where accuracy is misleading.
- **Threshold tuning:** Converting probabilistic outputs (e.g., from logistic regression) into binary decisions using precision-recall tradeoffs.
- **Production monitoring:** Tracking metric drift over time to detect when a deployed model degrades.
- **Stakeholder communication:** Translating technical metrics into business costs (e.g., "Our precision is 92%, so 8% of flagged transactions are false alarms").

## Resources

- [Scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) — Official documentation covering accuracy, precision, recall, F1, ROC AUC, and confusion matrices.
- [Scikit-learn: Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) — API reference for the confusion matrix, the foundation of all classification metrics.
- [Scikit-learn: Precision, Recall, F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) — The `classification_report` function you will use in practice.
- [Scikit-learn: ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) — Documentation for threshold-independent evaluation using the ROC curve.
- [Google Machine Learning Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification) — Visual intuition for precision, recall, and ROC AUC.

## Appendix

### Notation

- **TP** (True Positives): Correctly predicted positive cases.
- **FP** (False Positives): Negative cases incorrectly predicted as positive.
- **TN** (True Negatives): Correctly predicted negative cases.
- **FN** (False Negatives): Positive cases incorrectly predicted as negative.

### Core Metrics

**Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

The proportion of all predictions that were correct. Misleading when classes are imbalanced.

**Precision**

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all instances predicted as positive, how many were actually positive? High precision means few false alarms.

**Recall (Sensitivity / True Positive Rate)**

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual positive instances, how many did we catch? High recall means few missed positives.

**F1 Score**

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean of precision and recall. Use this when you need a single score that balances both.

**ROC AUC**

Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes across all possible thresholds. A score of 0.5 means random guessing; 1.0 means perfect separation.

**PR AUC**

Area Under the Precision-Recall curve. More informative than ROC AUC when dealing with highly imbalanced datasets.

### Common Pitfalls

- **Optimizing accuracy on imbalanced data:** A dummy classifier predicting the majority class can achieve high accuracy while being useless.
- **Ignoring the baseline:** Always compare your model's metrics against a naive baseline (e.g., always predict the majority class).
- **Using the wrong average for multiclass:** `micro`, `macro`, and `weighted` averages behave differently. Macro average treats all classes equally; weighted average accounts for class imbalance.
- **Threshold blindness:** Defaulting to a 0.5 probability threshold without considering the business cost of false positives vs. false negatives.

### Further Reading

- [Distill.pub: Machine Learning Research](https://distill.pub) — Interactive visual explanations of ML concepts.
- [Google Developers: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) — Visual guide to the precision-recall tradeoff.
