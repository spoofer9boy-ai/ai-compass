# Ordinary Least Squares

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 37 (Linear Regression)  
**Estimated Time:** 55 minutes

## Why am I learning this?

You already know that linear regression fits a line (or hyperplane) to data. In practice, "fitting" means choosing the parameters that minimize some error. Ordinary Least Squares (OLS) is the closed-form way to do that minimization. You will rarely write the normal equations by hand in production, but you will constantly debug why `np.linalg.lstsq` or `LinearRegression().fit()` gives a coefficient you did not expect. Understanding OLS means understanding where those numbers come from, when the math is exact, and when it silently breaks.

OLS is also the baseline against which every other regression method is compared. Ridge regression adds a penalty term. LASSO changes the penalty shape. Gradient descent replaces the closed-form solution with iteration. If you do not know the OLS target—minimize the sum of squared residuals—you cannot reason about why those variants were invented or when they outperform the original.

Finally, OLS is the gateway to statistical inference in ML. The same matrix algebra that gives you coefficients also gives you standard errors, confidence intervals, and the geometry of least-squares projection. Those ideas show up in A/B testing, causal inference, and even in understanding how attention heads project data in transformers.

## Where will I be using it?

- **Baseline modeling:** Before trying XGBoost or a neural net, fit OLS to establish a lower-bound metric. Data scientists at Stripe and Netflix still do this.
- **Causal inference:** OLS coefficients are interpreted as treatment effects in randomized experiments and observational studies with controls.
- **Feature importance:** The magnitude and sign of OLS coefficients provide a first-pass understanding of which features drive the target.
- **Residual analysis:** OLS residuals are used to detect heteroscedasticity, outliers, and model misspecification.
- **Signal processing:** Least-squares projection appears in filter design, system identification, and anywhere you project a signal onto a basis.

## Resources

- [scikit-learn: Linear Regression (OLS)](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) — The API you will actually call; includes complexity notes and caveats.
- [statsmodels OLS Documentation](https://www.statsmodels.org/stable/regression.html) — Deep statistical output: standard errors, t-tests, R-squared, and diagnostics.
- [Stanford CS229 Notes: Linear Regression](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Derives the normal equations from first principles with matrix notation.
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/lessons/neural-networks) — Visual intuition for why minimizing squared error projects data onto the column space.
- [CMU 36-707: Regression Lecture Notes](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf) — Rigorous treatment of the geometry of OLS, projection matrices, and the Gauss-Markov theorem.

## Appendix

### Notation

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: Design matrix with $n$ samples and $d$ features (often including a column of ones for the intercept).
- $\mathbf{y} \in \mathbb{R}^{n}$: Target vector.
- $\boldsymbol{\beta} \in \mathbb{R}^{d}$: Parameter vector we want to estimate.
- $\hat{\boldsymbol{\beta}}$: The OLS estimate of $\boldsymbol{\beta}$.

### The Normal Equations

OLS minimizes the residual sum of squares:

$$J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Taking the gradient with respect to $\boldsymbol{\beta}$ and setting it to zero:

$$\nabla_{\boldsymbol{\beta}} J = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$

This yields the normal equations:

$$\mathbf{X}^\top\mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$$

If $\mathbf{X}^\top\mathbf{X}$ is invertible, the unique solution is:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

### Geometric Interpretation

The predicted values $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$ are the orthogonal projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$. The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every column of $\mathbf{X}$, which is exactly what the normal equations encode: $\mathbf{X}^\top\mathbf{e} = \mathbf{0}$.

### Common Pitfalls

- **Multicollinearity:** If columns of $\mathbf{X}$ are linearly dependent, $\mathbf{X}^\top\mathbf{X}$ is singular and the inverse does not exist. In practice, this manifests as huge coefficient variances or numerical instability.
- **Outliers:** Squared error penalizes large residuals quadratically. A single outlier can pull the fit dramatically. Robust regression (e.g., Huber loss) exists for this reason.
- **Extrapolation:** OLS fits are only reliable within the convex hull of the training data. Predicting far outside that range is dangerous regardless of how good $R^2$ looks.
- **Non-linear relationships:** OLS assumes the conditional expectation of $y$ is linear in the parameters. If the true relationship is curved, residuals will show systematic structure.

### Further Reading

- [Wikipedia: Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) — Comprehensive reference on assumptions, properties, and extensions.
- [The Elements of Statistical Learning, Section 3.2](https://web.stanford.edu/~hastie/ElemStatLearn/) — Compares OLS to shrinkage methods in a unified framework.
