# ML Workflow Overview

**Phase:** PHASE-02-ml-core  
**Prerequisites:** None  
**Estimated Time:** 40 minutes

## Why am I learning this?

You will never build a machine learning model in a vacuum. In production, ML is a pipeline: raw data enters one end, and a deployed model serving predictions exits the other. If you do not understand how the stages connect, you will spend weeks debugging issues that are not model issues at all—they are workflow issues. A data leak from improper train-test splitting, a feature that cannot be computed in production, or a model that drifts silently because nobody set up monitoring: these are the failures that kill ML projects in industry.

This file exists to give you the map before you start driving. Every subject that follows—feature engineering, linear regression, cross-validation, regularization—sits inside this workflow. Understanding the big picture means you will know *when* to apply a technique, not just *how*.

## Where will I be using it?

- **Kaggle competitions:** Even solo projects follow a workflow—EDA, cleaning, modeling, ensembling. Knowing the stages keeps your notebooks organized.
- **Scikit-learn Pipelines:** `Pipeline` and `ColumnTransformer` are explicit workflow objects. You will use them to prevent data leakage and make your code reproducible.
- **MLOps platforms:** Tools like MLflow, Weights & Biases, and Databricks all assume you understand the standard ML lifecycle. Their UI is organized around it.
- **Team code reviews:** Senior engineers will ask why you split data before imputing, or whether your validation strategy matches production. This workflow is the vocabulary you need to answer.
- **Production debugging:** When model accuracy drops, the workflow tells you where to look first—data drift, feature pipeline bugs, or model staleness.

## Resources

- [ml-ops.org: End-to-End ML Workflow](https://ml-ops.org/content/end-to-end-ml-workflow) — High-level overview of data engineering, model engineering, and code engineering phases.
- [NVIDIA Developer Blog: Machine Learning in Practice](https://developer.nvidia.com/blog/machine-learning-in-practice-ml-workflows/) — Practical walkthrough of building an ML workflow with real-world constraints.
- [Scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html) — Official introduction to the scikit-learn API, including `fit`, `predict`, and `Pipeline`.
- [arXiv: Managing Machine Learning Workflow Components](https://arxiv.org/abs/1912.05665) — Academic survey of workflow component management and reproducibility challenges.
- [Databricks: MLOps Workflows](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow) — Industry-standard MLOps workflow documentation from a major platform.

## Appendix

### The Canonical ML Workflow

1. **Problem Definition** — What are you predicting? What is the success metric?
2. **Data Collection** — Gather raw data from databases, APIs, logs, or files.
3. **Exploratory Data Analysis (EDA)** — Understand distributions, correlations, and data quality issues.
4. **Data Preparation** — Cleaning, missing-value imputation, encoding, and scaling.
5. **Feature Engineering** — Create new features or transform existing ones to improve model signal.
6. **Train-Validation-Test Split** — Separate data *before* any transformation that uses global statistics.
7. **Model Selection & Training** — Try algorithms, tune hyperparameters, and compare metrics.
8. **Evaluation** — Assess performance on the held-out test set using task-appropriate metrics.
9. **Deployment** — Package the model, expose it via API or batch job, and integrate with production systems.
10. **Monitoring & Maintenance** — Track predictions, detect drift, and retrain on new data.

### Common Pitfalls

- **Data leakage:** Applying transformations (scaling, imputation) to the full dataset before splitting. Always fit on train, transform on validation/test.
- **Target leakage:** Including features that will not be available at prediction time in production.
- **Metric mismatch:** Optimizing for accuracy when the business cares about recall, precision, or revenue.
- **No reproducibility:** Random seeds, data versions, and environment dependencies must be tracked.

### Further Reading

- [Google Cloud: ML on GCP Best Practices](https://docs.cloud.google.com/architecture/ml-on-gcp-best-practices) — Cloud-native ML architecture guidance.
- [Distill.pub](https://distill.pub) — Interactive explanations of ML concepts if you want deeper intuition on any stage.
