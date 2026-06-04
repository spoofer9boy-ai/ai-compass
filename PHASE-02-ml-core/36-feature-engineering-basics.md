# Feature Engineering Basics

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 34 (ML Workflow Overview)  
**Estimated Time:** 60 minutes

## Why am I learning this?

You can have the most sophisticated model architecture in the world, but if your features are garbage, the model will learn garbage. In practice, the difference between a model that barely works and one that ships to production often comes down to how the raw data was cleaned, transformed, and encoded before it ever reached the estimator. Feature engineering is the bridge between the messy reality of business data and the mathematical assumptions of machine learning algorithms.

You will spend more time on feature engineering than on model selection in most real jobs. Kaggle competitions are frequently won not by using a fancier algorithm, but by constructing better features from the same raw data. Understanding the basic transformations — scaling, encoding, imputation, and binning — lets you turn a dataframe of strings, missing values, and wildly different numeric ranges into something a model can actually digest. This file covers the essential toolkit that every ML engineer uses daily.

## Where will I be using it?

- **Tabular Data Pipelines:** Preprocessing CSVs from databases before feeding them to scikit-learn, XGBoost, or neural networks.
- **Production ML Systems:** Building robust `Pipeline` objects that handle new data the same way as training data, preventing data leakage.
- **Recommendation Engines:** Encoding user IDs, item categories, and interaction counts into a numeric feature matrix.
- **Fraud Detection:** Creating interaction features and binning transaction amounts to capture non-linear patterns.
- **Time-Series Forecasting:** Extracting cyclical features (hour of day, day of week) from timestamps using custom transformers.

## Resources

- [scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html) — Official documentation covering scalers, encoders, and transformers.
- [scikit-learn: Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html) — How to handle `NaN`s with `SimpleImputer` and `IterativeImputer`.
- [scikit-learn: Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html) — Building safe, reproducible preprocessing pipelines with `Pipeline` and `ColumnTransformer`.
- [scikit-learn: Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html) — Removing low-variance or uninformative features before training.
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) — High-level intuition on transforming raw data into features.

## Appendix

### Notation

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: Data matrix with $n$ samples and $d$ features.
- $x_{ij}$: The value of the $j$-th feature for the $i$-th sample.
- $\mu_j$, $\sigma_j$: Mean and standard deviation of feature $j$.

### Common Pitfalls

- **Data leakage:** Fitting a scaler or imputer on the full dataset (including test data) before splitting. Always fit on train, transform on test.
- **Ordinal encoding for nominal categories:** Using `OrdinalEncoder` on unordered categories (like "color") makes the model assume an ordering that does not exist. Use `OneHotEncoder` instead.
- **Ignoring unseen categories:** A `OneHotEncoder` fit on training data will throw an error at inference time if a new category appears. Use `handle_unknown='ignore'` or `'infrequent_if_exist'`.
- **Scaling target variables:** Standardizing `y` is sometimes done in regression, but you must remember to invert the scaling on predictions before reporting results.

### Further Reading

- [scikit-learn: Compare the effect of different scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) — Visual guide to when to use `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.
- [scikit-learn: Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) — End-to-end example of preprocessing heterogeneous data.
