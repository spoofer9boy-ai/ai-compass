# Practice: Predict House Prices

**Phase:** PHASE-02-ml-core  
**Subjects Required:** 34 ML Workflow Overview, 36 Feature Engineering Basics, 37 Linear Regression, 38 Ordinary Least Squares  
**Estimated Time:** 180 minutes  
**Difficulty:** Intermediate

## Industry Context

You are the first data scientist at a mid-sized real-estate analytics company. Your product team wants a prototype that predicts house sale prices from public listing data so agents can flag underpriced properties. You have a CSV of 1,460 Ames, Iowa sales with 79 features, but the team has no ML infrastructure yet — you must build the entire pipeline in a single Jupyter-style script using only NumPy and pandas, with scikit-learn reserved strictly for the final train/test split and metric computation. The CTO wants to see *why* the model thinks a house is expensive, so you need interpretable coefficients, not a black box.

## The Problem

Build an end-to-end linear regression pipeline that predicts the final sale price (`SalePrice`) of a house given its features.

You must:

1. Load and inspect the data.
2. Handle missing values and encode categorical variables with a simple, reproducible strategy.
3. Split the data into training and validation sets (80/20).
4. Fit a linear regression model using **Ordinary Least Squares (closed-form normal equations)**.
5. Evaluate with RMSE and R² on the validation set.
6. Inspect the learned coefficients to identify the three features that most increase price and the three that most decrease it.

You are **not** required to use gradient descent here — OLS is explicitly the target subject. If you want a stretch goal, compare your OLS solution to a manual gradient-descent implementation and note the speed difference.

## Constraints

- Do **not** use `sklearn.linear_model.LinearRegression` or any other pre-built regression estimator. Implement OLS yourself with NumPy.
- You may use `sklearn.model_selection.train_test_split` and `sklearn.metrics.mean_squared_error` / `r2_score`.
- All feature engineering must be deterministic (no random imputation).
- The notebook/script must run on a single CPU core in under 5 seconds after data is loaded.
- Target validation RMSE < $50,000 and R² > 0.75 on the classic Ames Housing dataset.

## Starter Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
# Assume train.csv is the classic Kaggle "House Prices" dataset.
# If you don't have it locally, download from:
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
# For this exercise, we'll generate a tiny synthetic stand-in so the script runs.
# Replace this block with pd.read_csv("train.csv") in real use.
# ------------------------------------------------------------------

def make_synthetic_housing(n=1200):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "GrLivArea": rng.lognormal(7.5, 0.3, n),          # living area
        "OverallQual": rng.integers(1, 11, n),            # 1-10 quality rating
        "YearBuilt": rng.integers(1872, 2011, n),         # construction year
        "TotalBsmtSF": rng.lognormal(7.0, 0.5, n),        # basement sqft
        "GarageCars": rng.integers(0, 5, n),              # garage capacity
        "Neighborhood": rng.choice(["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"], n),
        "SalePrice": 0.0
    })
    # Synthetic price = linear combo + noise
    df["SalePrice"] = (
        50_000
        + 60 * df["GrLivArea"]
        + 12_000 * df["OverallQual"]
        + 300 * (df["YearBuilt"] - 1872)
        + 40 * df["TotalBsmtSF"]
        + 8_000 * df["GarageCars"]
        + df["Neighborhood"].map({"NoRidge": 40_000, "CollgCr": 15_000, "Veenker": 10_000, "Crawfor": 5_000, "Mitchel": 0})
        + rng.normal(0, 20_000, n)
    ).astype(int)
    return df

# df = pd.read_csv("train.csv")
df = make_synthetic_housing()

# ------------------------------------------------------------------
# 2. Preprocess
# ------------------------------------------------------------------
# TODO: Handle missing values (fill numeric with median, categorical with mode).
# TODO: Encode categoricals with one-hot encoding (pandas.get_dummies).
# TODO: Separate X and y. Drop non-predictive columns like Id if present.
# ------------------------------------------------------------------

# Placeholder — replace with real preprocessing
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"].values

# ------------------------------------------------------------------
# 3. Train / validation split
# ------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 4. OLS closed-form solution
# ------------------------------------------------------------------
# TODO: Convert X_train and X_val to NumPy matrices.
# TODO: Add a column of 1s for the intercept (bias term).
# TODO: Compute weights w = (X^T X)^(-1) X^T y  using np.linalg.pinv for stability.
# TODO: Predict on validation set.
# ------------------------------------------------------------------

# Placeholder predictions (random) so the script runs without crashing
val_preds = np.random.default_rng(0).integers(100_000, 300_000, size=len(y_val))

# ------------------------------------------------------------------
# 5. Evaluate
# ------------------------------------------------------------------
rmse = mean_squared_error(y_val, val_preds, squared=False)
r2 = r2_score(y_val, val_preds)
print(f"Validation RMSE: ${rmse:,.0f}")
print(f"Validation R²:   {r2:.3f}")

# ------------------------------------------------------------------
# 6. Inspect coefficients
# ------------------------------------------------------------------
# TODO: After computing w, map each weight back to its feature name.
# TODO: Print the top 3 positive and top 3 negative coefficients.
# ------------------------------------------------------------------
```

## Evaluation Criteria

1. **Correctness:** Validation RMSE < $50,000 and R² > 0.75 on the synthetic dataset (or equivalent thresholds on the real Ames dataset).
2. **OLS Implementation:** You explicitly compute `w = (XᵀX)⁻¹ Xᵀy` (or the pseudo-inverse variant) — no `sklearn.linear_model`.
3. **Preprocessing:** Missing values and categoricals are handled deterministically; the pipeline does not crash on unseen categories in validation (hint: fit dummies on training data only, then reindex validation).
4. **Interpretability:** You print the six most influential coefficients with their feature names, proving the model is inspectable.
5. **Edge handling:** Script runs without errors even if the dataset has zero rows or all-missing columns (defensive coding).

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------
# 1. Load data (synthetic stand-in)
# ------------------------------------------------------------------
def make_synthetic_housing(n=1200):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "GrLivArea": rng.lognormal(7.5, 0.3, n),
        "OverallQual": rng.integers(1, 11, n),
        "YearBuilt": rng.integers(1872, 2011, n),
        "TotalBsmtSF": rng.lognormal(7.0, 0.5, n),
        "GarageCars": rng.integers(0, 5, n),
        "Neighborhood": rng.choice(["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"], n),
        "SalePrice": 0.0
    })
    df["SalePrice"] = (
        50_000
        + 60 * df["GrLivArea"]
        + 12_000 * df["OverallQual"]
        + 300 * (df["YearBuilt"] - 1872)
        + 40 * df["TotalBsmtSF"]
        + 8_000 * df["GarageCars"]
        + df["Neighborhood"].map({"NoRidge": 40_000, "CollgCr": 15_000, "Veenker": 10_000, "Crawfor": 5_000, "Mitchel": 0})
        + rng.normal(0, 20_000, n)
    ).astype(int)
    return df

df = make_synthetic_housing()

# ------------------------------------------------------------------
# 2. Preprocess
# ------------------------------------------------------------------
# Separate target
y = df["SalePrice"].values
X_raw = df.drop(columns=["SalePrice"])

# Numeric vs categorical
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill missing values deterministically
X_proc = X_raw.copy()
X_proc[numeric_cols] = X_proc[numeric_cols].fillna(X_proc[numeric_cols].median())
X_proc[categorical_cols] = X_proc[categorical_cols].fillna(X_proc[categorical_cols].mode().iloc[0])

# One-hot encode categoricals
X_dummies = pd.get_dummies(X_proc, columns=categorical_cols, drop_first=False)

# ------------------------------------------------------------------
# 3. Train / validation split
# ------------------------------------------------------------------
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_dummies, y, test_size=0.2, random_state=42
)

# Align columns: validation must have exactly the same dummy columns as training
X_val_df = X_val_df.reindex(columns=X_train_df.columns, fill_value=0)

# Convert to NumPy
X_train = X_train_df.values.astype(float)
X_val = X_val_df.values.astype(float)

# Add intercept column (column of 1s)
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_val_b = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

# ------------------------------------------------------------------
# 4. OLS closed-form solution
# ------------------------------------------------------------------
# w = (X^T X)^(-1) X^T y   -> use pinv for numerical stability
w = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

# Predict
val_preds = X_val_b @ w

# ------------------------------------------------------------------
# 5. Evaluate
# ------------------------------------------------------------------
rmse = mean_squared_error(y_val, val_preds, squared=False)
r2 = r2_score(y_val, val_preds)
print(f"Validation RMSE: ${rmse:,.0f}")
print(f"Validation R²:   {r2:.3f}")

# ------------------------------------------------------------------
# 6. Inspect coefficients
# ------------------------------------------------------------------
feature_names = ["Intercept"] + X_train_df.columns.tolist()
coefs = pd.Series(w, index=feature_names)

top_positive = coefs.sort_values(ascending=False).head(3)
top_negative = coefs.sort_values().head(3)

print("\nTop 3 price-increasing features:")
for name, value in top_positive.items():
    print(f"  {name}: +${value:,.0f}")

print("\nTop 3 price-decreasing features:")
for name, value in top_negative.items():
    print(f"  {name}: ${value:,.0f}")
```

**Why this works:**
- **Matrix Multiplication (Subject 4):** `X_train_b.T @ X_train_b` builds the Gram matrix required by the normal equations.
- **Matrix Inverse / Pseudo-inverse (Subject 6):** `np.linalg.pinv` computes the Moore-Penrose inverse, which is numerically safer than a raw inverse when features are correlated or nearly singular.
- **Linear Regression (Subject 37) & OLS (Subject 38):** The entire solution is the closed-form OLS estimator $\hat{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$.
- **Feature Engineering (Subject 36):** Missing-value imputation and one-hot encoding are done deterministically; validation columns are reindexed to prevent shape mismatches.

</details>

## What You Actually Learned

- **ML Workflow Overview:** You walked through the full pipeline — load, clean, split, train, evaluate, interpret — in one script.
- **Feature Engineering Basics:** You handled missing values, encoded categoricals, and aligned train/validation schemas so the model doesn't crash on unseen categories.
- **Linear Regression:** You connected the abstract math (a linear model) to a concrete business metric (dollars of pricing error).
- **Ordinary Least Squares:** You implemented the normal equations directly with NumPy, giving you an exact solution in one shot and a clear view of how every feature influences price.

This is the same analytical pattern used by Zillow's Zestimate, Redfin's estimate models, and countless internal pricing tools — only those systems add regularization, spatial features, and ensemble layers on top of this foundation.
