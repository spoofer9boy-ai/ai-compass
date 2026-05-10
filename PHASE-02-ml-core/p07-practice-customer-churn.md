# Practice: Customer Churn Prediction

**Phase:** PHASE-02-ml-core  
**Subjects Required:** 40 (Train-Val-Test Split), 41 (Feature Engineering Basics), 48 (Cross-Validation)  
**Estimated Time:** 150 minutes  
**Difficulty:** Intermediate

## Industry Context

You are the first ML engineer at a subscription-based SaaS startup. The customer success team is flying blind: they only know a customer has left after the cancellation email arrives. The CEO wants a model that flags at-risk customers *before* they churn so the team can intervene with retention offers. You have a CSV of 7,043 customer records with demographic and account features, but no dedicated ML platform — you must build the entire pipeline in a single script using scikit-learn for modeling and evaluation, with a focus on **interpretable classification metrics** that the business can act on.

## The Problem

Build an end-to-end binary classification pipeline that predicts whether a customer will churn (`Churn = Yes/No`).

You must:

1. Load and inspect the data.
2. Handle missing values, encode categorical variables, and scale numeric features with a reproducible strategy.
3. Split the data into training and validation sets (80/20) using a stratified split to preserve class balance.
4. Train a **Logistic Regression** classifier and a **Random Forest** classifier.
5. Evaluate both models with **precision, recall, F1-score, and ROC-AUC** on the validation set.
6. Perform **5-fold stratified cross-validation** on the training set to estimate generalization stability.
7. Identify the top 5 features driving churn according to the Random Forest model and explain why they matter to the business.

You are **not** required to implement logistic regression or random forest from scratch — scikit-learn estimators are explicitly allowed here. The target subjects are the *evaluation protocol* (train/val split, cross-validation) and the *feature engineering* that makes the models reliable, not the algorithm internals.

## Constraints

- You may use `pandas`, `numpy`, and `scikit-learn` only. No XGBoost, LightGBM, or deep-learning frameworks.
- The train/validation split must be **stratified** (`stratify=y`) because churn datasets are typically imbalanced (~25–30% churn).
- Cross-validation must be **StratifiedKFold** with 5 folds, shuffled, `random_state=42`.
- All preprocessing (imputation, encoding, scaling) must be fit on the training data only and applied to validation — no data leakage.
- The script must run on a single CPU core in under 5 seconds after data is loaded.
- Target validation F1-score > 0.60 and ROC-AUC > 0.75 on the classic Telco Customer Churn dataset.

## Starter Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support
)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
# Assume WA_Fn-UseC_-Telco-Customer-Churn.csv is the classic IBM dataset.
# If you don't have it locally, download from:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn (requires Kaggle account)
# For this exercise, we'll generate a tiny synthetic stand-in so the script runs.
# Replace this block with pd.read_csv("...") in real use.
# ------------------------------------------------------------------

def make_synthetic_churn(n=2000):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "tenure": rng.integers(0, 73, n),
        "MonthlyCharges": rng.normal(65, 30, n).round(2),
        "TotalCharges": rng.normal(2283, 2000, n).round(2),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.30, 0.15]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
        ),
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "Churn": "No"
    })
    # Synthetic churn logic: higher churn for short tenure + high monthly + month-to-month
    churn_prob = (
        0.10
        + 0.25 * (df["tenure"] < 12).astype(float)
        + 0.20 * (df["Contract"] == "Month-to-month").astype(float)
        + 0.15 * (df["InternetService"] == "Fiber optic").astype(float)
        + 0.10 * (df["PaymentMethod"] == "Electronic check").astype(float)
        + 0.05 * (df["MonthlyCharges"] > 80).astype(float)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df["Churn"] = rng.random(n) < churn_prob
    df["Churn"] = df["Churn"].map({True: "Yes", False: "No"})
    # Inject some missing TotalCharges for realism
    missing_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan
    return df

# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = make_synthetic_churn()

# ------------------------------------------------------------------
# 2. Preprocess
# ------------------------------------------------------------------
# TODO: Separate X and y. Target column is "Churn".
# TODO: Identify numeric and categorical columns.
# TODO: Build a ColumnTransformer that:
#       - imputes numeric missing values with median and scales with StandardScaler
#       - imputes categorical missing values with "missing" and one-hot encodes
# TODO: Fit the transformer on training data only (see split below).
# ------------------------------------------------------------------

# Placeholder — replace with real preprocessing
X = df.drop(columns=["Churn"])
y = df["Churn"].values

# ------------------------------------------------------------------
# 3. Train / validation split (stratified!)
# ------------------------------------------------------------------
# TODO: Use train_test_split with stratify=y, test_size=0.2, random_state=42.
# ------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # FIXME: add stratify=y

# ------------------------------------------------------------------
# 4. Build preprocessing pipeline
# ------------------------------------------------------------------
# TODO: Create ColumnPreprocessor and fit on X_train.
# TODO: Transform X_train and X_val.
# ------------------------------------------------------------------

# Placeholder — no-op so the script runs
X_train_proc = X_train.select_dtypes(include=[np.number]).fillna(0).values
X_val_proc = X_val.select_dtypes(include=[np.number]).fillna(0).values

# ------------------------------------------------------------------
# 5. Train models
# ------------------------------------------------------------------
# TODO: Train LogisticRegression(max_iter=1000, random_state=42)
# TODO: Train RandomForestClassifier(n_estimators=200, random_state=42)
# ------------------------------------------------------------------

log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)

log_reg.fit(X_train_proc, y_train)
rf_clf.fit(X_train_proc, y_train)

# ------------------------------------------------------------------
# 6. Evaluate on validation set
# ------------------------------------------------------------------
# TODO: Predict labels and probabilities on X_val_proc.
# TODO: Print classification_report for both models.
# TODO: Print ROC-AUC for both models (use predict_proba[:, 1] and map "Yes"->1).
# ------------------------------------------------------------------

log_pred = log_reg.predict(X_val_proc)
rf_pred = rf_clf.predict(X_val_proc)

print("Logistic Regression report:")
print(classification_report(y_val, log_pred))

print("Random Forest report:")
print(classification_report(y_val, rf_pred))

# ------------------------------------------------------------------
# 7. Cross-validation on training set
# ------------------------------------------------------------------
# TODO: Use StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO: Run cross_val_score with scoring="roc_auc" for both models on X_train_proc, y_train.
# TODO: Print mean ± std for each model.
# ------------------------------------------------------------------

# Placeholder
print("CV mean ROC-AUC: 0.500 ± 0.00 (placeholder)")

# ------------------------------------------------------------------
# 8. Feature importance
# ------------------------------------------------------------------
# TODO: Extract feature_importances_ from the trained Random Forest.
# TODO: Map them back to original feature names (accounting for one-hot expansion).
# TODO: Print top 5 features that drive churn.
# ------------------------------------------------------------------

print("Top 5 churn drivers: (placeholder)")
```

## Evaluation Criteria

1. **Correctness:** Validation F1-score > 0.60 and ROC-AUC > 0.75 on the synthetic dataset (or equivalent thresholds on the real Telco dataset).
2. **Stratified Split:** `train_test_split` uses `stratify=y` so the validation set preserves the churn rate.
3. **No Data Leakage:** Imputers and scalers are fit on `X_train` only; `X_val` is transformed, never used for fitting.
4. **Cross-Validation:** 5-fold stratified CV reports mean ± std ROC-AUC, proving the model is stable across data folds.
5. **Interpretability:** You print the top 5 Random Forest feature importances with human-readable names, and explain in 1–2 sentences why each matters to customer retention.
6. **Edge Handling:** Script handles missing `TotalCharges`, unseen categories in validation (via `handle_unknown="ignore"` in OneHotEncoder), and runs without errors on empty or all-missing columns.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ------------------------------------------------------------------
# 1. Load data (synthetic stand-in)
# ------------------------------------------------------------------
def make_synthetic_churn(n=2000):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "tenure": rng.integers(0, 73, n),
        "MonthlyCharges": rng.normal(65, 30, n).round(2),
        "TotalCharges": rng.normal(2283, 2000, n).round(2),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.30, 0.15]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
        ),
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "Churn": "No"
    })
    churn_prob = (
        0.10
        + 0.25 * (df["tenure"] < 12).astype(float)
        + 0.20 * (df["Contract"] == "Month-to-month").astype(float)
        + 0.15 * (df["InternetService"] == "Fiber optic").astype(float)
        + 0.10 * (df["PaymentMethod"] == "Electronic check").astype(float)
        + 0.05 * (df["MonthlyCharges"] > 80).astype(float)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df["Churn"] = rng.random(n) < churn_prob
    df["Churn"] = df["Churn"].map({True: "Yes", False: "No"})
    missing_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan
    return df

df = make_synthetic_churn()

# ------------------------------------------------------------------
# 2. Preprocess
# ------------------------------------------------------------------
y = df["Churn"].map({"Yes": 1, "No": 0}).values
X = df.drop(columns=["Churn"])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols),
    ]
)

# ------------------------------------------------------------------
# 3. Train / validation split (stratified)
# ------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit preprocessor on training data only
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

# ------------------------------------------------------------------
# 4. Train models
# ------------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)

log_reg.fit(X_train_proc, y_train)
rf_clf.fit(X_train_proc, y_train)

# ------------------------------------------------------------------
# 5. Evaluate on validation set
# ------------------------------------------------------------------
def evaluate_model(model, name):
    preds = model.predict(X_val_proc)
    probs = model.predict_proba(X_val_proc)[:, 1]
    print(f"\n=== {name} ===")
    print(classification_report(y_val, preds, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_val, probs):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_val, preds))

evaluate_model(log_reg, "Logistic Regression")
evaluate_model(rf_clf, "Random Forest")

# ------------------------------------------------------------------
# 6. Cross-validation on training set
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_cv = cross_val_score(log_reg, X_train_proc, y_train, cv=cv, scoring="roc_auc")
rf_cv = cross_val_score(rf_clf, X_train_proc, y_train, cv=cv, scoring="roc_auc")

print(f"\nLogistic Regression CV ROC-AUC: {log_cv.mean():.3f} ± {log_cv.std():.3f}")
print(f"Random Forest CV ROC-AUC:       {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

# ------------------------------------------------------------------
# 7. Feature importance
# ------------------------------------------------------------------
# Get feature names after one-hot encoding
feature_names = (
    numeric_cols +
    list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols))
)

importances = pd.Series(rf_clf.feature_importances_, index=feature_names)
top5 = importances.sort_values(ascending=False).head(5)

print("\nTop 5 churn drivers (Random Forest):")
for feat, imp in top5.items():
    print(f"  {feat}: {imp:.4f}")

print("""
Business interpretation:
- tenure: New customers (< 12 months) are far more likely to leave.
- Contract (Month-to-month): No long-term commitment means low switching cost.
- InternetService (Fiber optic): Higher price point may drive price-sensitive customers away.
- PaymentMethod (Electronic check): Often correlates with lower auto-pay adoption / higher friction.
- MonthlyCharges: Higher bills increase the incentive to shop for cheaper alternatives.
""")
```

**Why this works:**
- **Train-Val-Test Split (Subject 40):** The stratified 80/20 split guarantees that the validation set mirrors the real-world churn rate (~27% in the Telco dataset), preventing overly optimistic metrics on a validation set that accidentally contains too many churners or non-churners.
- **Feature Engineering Basics (Subject 41):** Missing `TotalCharges` is imputed with the median (fit on training only); categoricals are one-hot encoded with `handle_unknown="ignore"` so unseen categories in validation don't crash the pipeline; numeric features are standardized so logistic regression converges faster.
- **Cross-Validation (Subject 48):** 5-fold stratified CV gives a robust estimate of how the model will generalize to new customer cohorts. If the mean ROC-AUC is 0.78 ± 0.02, you can confidently tell the CEO the model will perform consistently; if it's 0.78 ± 0.08, the model is unstable and needs more data or simpler features.
- **Classification Metrics:** Precision tells the retention team "of the customers we flagged, how many actually left?" Recall tells them "of the customers who left, how many did we catch?" F1 balances the two, and ROC-AUC measures the model's ability to rank churners above non-churners — critical when the team has limited outreach capacity and must prioritize the highest-risk accounts.

</details>

## What You Actually Learned

- **Train-Val-Test Split:** You used a stratified split to preserve the natural class imbalance, ensuring your evaluation reflects real-world performance instead of a lucky random partition.
- **Feature Engineering Basics:** You built a reusable `ColumnTransformer` pipeline that handles mixed data types, prevents data leakage, and survives unseen categories in production — the exact pattern used in every scikit-learn deployment.
- **Cross-Validation:** You quantified model stability with mean ± std across folds, giving the business a confidence interval rather than a single point estimate. If the std is high, you know the model overfits to specific customer segments.
- **Classification Metrics:** You moved beyond "accuracy" (which is misleading on imbalanced data) to precision, recall, F1, and ROC-AUC — the metrics every product team actually asks for when the cost of false negatives (missing a churner) differs from the cost of false positives (wasting a retention offer).

This is the same analytical pattern used by telecom retention teams at Verizon, SaaS growth teams at HubSpot, and subscription analytics at Netflix — only those systems add automated retraining, feature stores, and causal uplift modeling on top of this foundation.

## Resources

- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — The API you will actually use for linear classification.
- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) — Ensemble trees with built-in feature importance.
- [scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) — Precision, recall, F1, ROC-AUC, and when to use each.
- [Google ML Crash Course: Classification Metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) — Visual intuition for precision/recall trade-offs on imbalanced data.
- [Wikipedia: Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) — Formal definitions and the confusion matrix decomposition.
