# Logistic Regression

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 37 (Linear Regression), 38 (Ordinary Least Squares)  
**Estimated Time:** 60 minutes

## Why am I learning this?

Linear regression predicts continuous values, but most business problems are classification problems: will this customer churn? Is this transaction fraudulent? Does this email contain malware? You cannot solve these with a straight line that outputs unbounded real numbers. You need a model that outputs probabilities bounded between 0 and 1, and that is exactly what logistic regression provides.

Despite its name, logistic regression is a classification algorithm. It wraps a linear model inside a sigmoid function, squashing the output into a valid probability. The model assumes that the log-odds of the positive class is a linear combination of the input features. This assumption is restrictive — it means the decision boundary is a hyperplane — but that restriction is also its strength. Logistic regression is interpretable, fast to train, and surprisingly hard to beat as a baseline. In many production systems at banks, insurers, and healthcare companies, logistic regression remains the deployed model because regulators and auditors can inspect its coefficients and understand exactly why a decision was made.

The training procedure also teaches you a foundational concept: maximum likelihood estimation. Unlike linear regression, which has a closed-form solution via the normal equations, logistic regression has no closed form. You minimize the negative log-likelihood — also called cross-entropy loss — using gradient descent. This is the exact same loss function and optimization algorithm that trains the final layer of every neural network classifier. If you understand logistic regression, you understand how a neural network learns to classify.

## Where will I be using it?

- **Baseline classification:** Before you try XGBoost or a neural net, fit logistic regression. If it achieves 90% of the performance with 1% of the compute, the complex model may not be worth the operational cost.
- **Credit scoring and risk modeling:** Banks use logistic regression to estimate default probability because regulators require interpretable models. The coefficient for "debt-to-income ratio" has a direct, auditable meaning.
- **Medical diagnosis:** Predicting disease presence from biomarkers. The odds ratios derived from logistic coefficients are standard reporting in clinical research.
- **Click-through rate prediction:** Ad-tech systems use logistic regression as a fast, online-learnable model to estimate the probability that a user clicks an ad.
- **Neural network final layers:** A binary classifier's output layer is mathematically equivalent to logistic regression. The `BCEWithLogitsLoss` in PyTorch is logistic regression's loss function.
- **A/B test analysis:** Estimating treatment effects on binary outcomes (conversion, retention) using logistic regression with covariate adjustment.

## Resources

- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) — The production API. Covers solvers, regularization, and multiclass strategies.
- [PyTorch Docs: BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — The numerically stable loss function used for binary classification in deep learning.
- [Wikipedia: Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) — Rigorous derivation of the model, likelihood function, and connection to generalized linear models.
- [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Derives logistic regression from maximum likelihood and connects it to the exponential family.
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks) — Visual intuition for sigmoid functions and how they convert scores to probabilities.

## Appendix

### Notation

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: Design matrix with $n$ samples and $d$ features.
- $\mathbf{y} \in \{0, 1\}^n$: Binary target vector.
- $\boldsymbol{\beta} \in \mathbb{R}^d$: Parameter vector (weights).
- $\sigma(z) = \frac{1}{1 + e^{-z}}$: The standard logistic (sigmoid) function.
- $\hat{p}_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$: Predicted probability that $y_i = 1$.

### The Model

Logistic regression models the probability of the positive class as:

$$
\Pr(Y_i = 1 \mid \mathbf{X}_i) = \sigma(\boldsymbol{\beta}^\top \mathbf{X}_i) = \frac{1}{1 + e^{-\boldsymbol{\beta}^\top \mathbf{X}_i}}
$$

The log-odds (logit) is linear in the features:

$$
\ln\left(\frac{\hat{p}_i}{1 - \hat{p}_i}\right) = \boldsymbol{\beta}^\top \mathbf{X}_i
$$

### The Loss Function

Parameters are estimated by maximizing the likelihood of the observed data, which is equivalent to minimizing the negative log-likelihood — also known as binary cross-entropy:

$$
J(\boldsymbol{\beta}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]
$$

There is no closed-form solution for the optimal $\boldsymbol{\beta}$. The gradient is:

$$
\nabla_{\boldsymbol{\beta}} J = \frac{1}{n} \mathbf{X}^\top (\hat{\mathbf{p}} - \mathbf{y})
$$

And gradient descent updates the parameters iteratively:

$$
\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t - \eta \nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}_t)
$$

### Common Pitfalls

- **Calling it regression:** It is a classifier. The "regression" in the name refers to the linear predictor, not the output type.
- **Ignoring class imbalance:** When 99% of labels are 0, the model can achieve 99% accuracy by always predicting 0. Use class weights, `pos_weight` in PyTorch, or metrics like AUC-ROC instead of raw accuracy.
- **Interpreting coefficients as linear effects:** A one-unit change in $x_j$ changes the *log-odds* by $\beta_j$, not the probability. The effect on probability depends on the current value of all features.
- **Perfect separation:** If a feature perfectly separates the classes, the likelihood has no maximum — coefficients diverge to infinity. Regularization or Firth's bias-reduced logistic regression fixes this.
- **Applying linear regression assumptions:** Logistic regression does not assume Gaussian residuals or homoscedasticity. It assumes independent observations and a linear relationship in log-odds.

### Further Reading

- [Elements of Statistical Learning, Section 4.4](https://web.stanford.edu/~hastie/ElemStatLearn/) — Logistic regression in the context of linear methods for classification.
- [Distill.pub: Visual Information Theory](https://distill.pub/2017/information-theory/) — Builds intuition for cross-entropy, the core loss function of logistic regression.
