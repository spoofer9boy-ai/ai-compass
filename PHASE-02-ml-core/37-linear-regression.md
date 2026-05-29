# Linear Regression

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 4 (Matrix Multiplication), 16 (Derivatives), 34 (ML Workflow Overview)  
**Estimated Time:** 60 minutes

## Why am I learning this?

Linear regression is the first predictive model most engineers build, and it is the conceptual ancestor of every neural network layer you will ever train. You will rarely ship a raw OLS model in production, but you will spend countless hours debugging why a deep network is not converging, and the debugging toolkit starts here: understanding residuals, coefficient scales, and the geometry of a loss surface.

The idea is simple. You have a matrix of observations $\mathbf{X}$ and a vector of targets $\mathbf{y}$. You assume the relationship between them is approximately linear, corrupted by some noise $\boldsymbol{\varepsilon}$:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

Your job is to estimate $\boldsymbol{\beta}$, the vector of weights that best explains the data. "Best" usually means minimizing the sum of squared errors — a choice that is mathematically convenient, statistically justified under Gaussian noise, and geometrically equivalent to projecting $\mathbf{y}$ onto the column space of $\mathbf{X}$.

Linear regression also teaches you how to think about the bias-variance tradeoff before you ever encounter those words. When features are correlated, the design matrix becomes close to singular and small changes in the data produce wild swings in the estimated weights. That fragility is why we later add regularization (ridge, lasso), but you cannot appreciate the fix until you feel the pain of the original problem.

## Where will I be using it?

- **Baseline modeling:** Before you reach for XGBoost or a transformer, you fit a linear model to know how much signal is actually in the features. If a linear model gets within 5% of a neural net, the neural net may not be worth the compute.
- **Deep learning internals:** Every `torch.nn.Linear` layer is a linear regression with learned weights. Understanding initialization scales, gradient flow, and output variance starts with the linear case.
- **Causal inference and econometrics:** Linear regression is still the workhorse for estimating treatment effects, difference-in-differences, and instrumental variables.
- **A/B testing and forecasting:** Predicting revenue from ad spend, estimating lift from a feature rollout, or decomposing seasonality often reduces to a linear model with engineered features.
- **Interpretability:** Linear coefficients have direct marginal interpretations. A one-unit change in feature $j$ changes the prediction by $\beta_j$, holding other features constant. That transparency is rare in non-linear models.

## Resources

- [Stanford CS229 Lecture Notes — Linear Regression](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Derives the normal equations and the probabilistic interpretation from first principles.
- [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) — The API you will actually call. Covers OLS, ridge, lasso, and practical pitfalls.
- [PyTorch Docs: torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) — The building block of neural networks. Understand what `in_features`, `out_features`, and `bias` mean geometrically.
- [Wikipedia: Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) — Surprisingly rigorous reference for notation, assumptions, and the matrix formulation.
- [fast.ai: Computational Linear Algebra](https://www.fast.ai/posts/2017-07-17-num-lin-alg.html) — Application-first perspective on why matrix operations matter for real data.

## Appendix

### Notation

- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$: Design matrix with $n$ samples and $p$ features, typically with a column of ones prepended for the intercept.
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$: Parameter vector (weights + intercept).
- $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$: Predicted values.
- $\boldsymbol{\varepsilon} = \mathbf{y} - \hat{\mathbf{y}}$: Residual vector.

### The Normal Equations

If $\mathbf{X}^T\mathbf{X}$ is invertible, the least-squares solution is:

$$
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

This is derived by setting the gradient of the squared-error loss to zero — a direct application of the calculus prerequisites.

### Common Pitfalls

- **Multicollinearity:** When two or more features are linearly dependent, $\mathbf{X}^T\mathbf{X}$ is singular and the normal equations break down. In practice, scikit-learn handles this via SVD, but coefficient variance inflates.
- **Extrapolation:** A linear model fit on $x \in [0, 10]$ has no guarantee of being accurate at $x = 1000$.
- **Confounding:** A coefficient $\beta_j$ measures association, not causation. If an unobserved variable influences both $x_j$ and $y$, the estimate is biased.

### Further Reading

- [Elements of Statistical Learning, Chapter 3](https://web.stanford.edu/~hastie/ElemStatLearn/) — Rigorous treatment of linear methods for regression and classification.
- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — Not linear regression specifically, but builds the geometric intuition that carries over.
