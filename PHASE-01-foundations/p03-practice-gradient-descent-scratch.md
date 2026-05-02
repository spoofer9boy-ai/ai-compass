# Practice: Gradient Descent from Scratch

**Phase:** PHASE-01-foundations  
**Subjects Required:** 16 (Derivatives), 17 (Partial Derivatives), 18 (Gradient Vector)  
**Estimated Time:** 150 minutes  
**Difficulty:** Intermediate

## Industry Context

You are an ML engineer at a health-tech startup building a wearable glucose predictor. The team wants to understand why the model converges slowly on-device. The embedded team cannot run PyTorch; they need a bare-metal NumPy implementation of the optimizer so they can profile memory and step-by-step convergence. Your job is to implement gradient descent from scratch, validate it against a known regression problem, and produce convergence diagnostics that the firmware team can replicate.

## The Problem

Implement **batch gradient descent** and **stochastic gradient descent (SGD)** from scratch in NumPy to fit a multivariate linear regression model. You must:

1. Generate a synthetic dataset with a known ground-truth weight vector.
2. Implement a mean squared error (MSE) cost function.
3. Compute gradients using only partial derivatives (no autograd).
4. Implement batch GD and SGD update rules.
5. Track loss per epoch and produce a convergence plot.
6. Compare the final learned weights to the ground truth.

You may **not** use `scikit-learn`, `torch.optim`, or any automatic differentiation library. NumPy and Matplotlib only.

## Constraints

- Implement gradient descent **from scratch** using only NumPy.
- Do **not** use autograd, `torch`, `tensorflow`, or `scikit-learn`.
- Dataset: 1,000 samples, 5 features, Gaussian noise (σ = 2.0).
- Ground-truth weights: `w_true = [2.0, -1.0, 0.5, 3.0, -2.0]`, bias `b_true = 1.0`.
- Learning rate: start at `0.01`. You may tune it, but document your choice.
- Epochs: 1,000 for batch GD; 100 epochs with batch size 32 for SGD.
- Must run on a single CPU core in under 5 seconds.

## Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Generate synthetic data
# ------------------------------------------------------------------
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
w_true = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
b_true = 1.0
noise = np.random.randn(n_samples) * 2.0
y = X @ w_true + b_true + noise

# ------------------------------------------------------------------
# 2. Cost function
# ------------------------------------------------------------------
def mse_cost(X, y, w, b):
    """
    Compute mean squared error.
    X: (n_samples, n_features)
    y: (n_samples,)
    w: (n_features,)
    b: scalar
    """
    # TODO: implement MSE
    pass

# ------------------------------------------------------------------
# 3. Gradient computation
# ------------------------------------------------------------------
def compute_gradients(X, y, w, b):
    """
    Compute partial derivatives of MSE with respect to w and b.
    Returns: dw (n_features,), db (scalar)
    """
    # TODO: implement gradient using partial derivatives
    pass

# ------------------------------------------------------------------
# 4. Batch Gradient Descent
# ------------------------------------------------------------------
def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    """
    Run batch gradient descent.
    Returns: w, b, loss_history (list)
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        # TODO: compute gradients and update parameters
        pass

    return w, b, loss_history

# ------------------------------------------------------------------
# 5. Stochastic Gradient Descent
# ------------------------------------------------------------------
def stochastic_gradient_descent(X, y, lr=0.01, epochs=100, batch_size=32):
    """
    Run SGD with mini-batches.
    Returns: w, b, loss_history (list of epoch-mean losses)
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        # TODO: shuffle data, iterate mini-batches, update parameters
        pass

    return w, b, loss_history

# ------------------------------------------------------------------
# 6. Run and compare
# ------------------------------------------------------------------
w_bg, b_bg, loss_bg = batch_gradient_descent(X, y)
w_sg, b_sg, loss_sg = stochastic_gradient_descent(X, y)

print("Batch GD   w:", w_bg.round(3), "b:", round(b_bg, 3))
print("SGD        w:", w_sg.round(3), "b:", round(b_sg, 3))
print("Ground truth w:", w_true, "b:", b_true)

# TODO: plot loss_bg and loss_sg on the same axes
```

## Evaluation Criteria

1. **Correctness:** Final `w` and `b` are within `0.3` of ground truth for batch GD and within `0.5` for SGD.
2. **Efficiency:** Full script runs in under 5 seconds on a single CPU core.
3. **Gradient correctness:** Analytical gradient matches a finite-difference check to within `1e-4`.
4. **Convergence:** Loss decreases monotonically (with SGD noise allowed) and plateaus below `5.0` MSE.
5. **Code clarity:** Comments explain the link to derivatives, partial derivatives, and the gradient vector.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Generate synthetic data
# ------------------------------------------------------------------
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
w_true = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
b_true = 1.0
noise = np.random.randn(n_samples) * 2.0
y = X @ w_true + b_true + noise

# ------------------------------------------------------------------
# 2. Cost function
# ------------------------------------------------------------------
def mse_cost(X, y, w, b):
    """
    Compute mean squared error.
    """
    predictions = X @ w + b
    errors = predictions - y
    return np.mean(errors ** 2)

# ------------------------------------------------------------------
# 3. Gradient computation
# ------------------------------------------------------------------
def compute_gradients(X, y, w, b):
    """
    Compute partial derivatives of MSE with respect to w and b.

    MSE = (1/n) * sum((Xw + b - y)^2)
    d(MSE)/dw_j = (2/n) * sum((Xw + b - y) * X_j)   <-- partial derivative
    d(MSE)/db   = (2/n) * sum((Xw + b - y))         <-- partial derivative

    The gradient vector ∇L is the collection of all partial derivatives.
    """
    n_samples = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    dw = (2.0 / n_samples) * (X.T @ errors)
    db = (2.0 / n_samples) * np.sum(errors)
    return dw, db

# ------------------------------------------------------------------
# 4. Batch Gradient Descent
# ------------------------------------------------------------------
def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    """
    Batch GD: uses the full dataset to compute the exact gradient each step.
    The gradient vector points in the direction of steepest ascent,
    so we subtract it (times learning rate) to descend.
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        loss = mse_cost(X, y, w, b)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db

    return w, b, loss_history

# ------------------------------------------------------------------
# 5. Stochastic Gradient Descent
# ------------------------------------------------------------------
def stochastic_gradient_descent(X, y, lr=0.01, epochs=100, batch_size=32):
    """
    SGD: approximates the true gradient using a mini-batch.
    Each step uses partial derivatives computed on a subset,
    introducing noise that can help escape shallow local minima.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        # Shuffle indices for unbiased mini-batch sampling
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            loss = mse_cost(X_batch, y_batch, w, b)
            epoch_losses.append(loss)

            dw, db = compute_gradients(X_batch, y_batch, w, b)
            w -= lr * dw
            b -= lr * db

        loss_history.append(np.mean(epoch_losses))

    return w, b, loss_history

# ------------------------------------------------------------------
# 6. Run and compare
# ------------------------------------------------------------------
w_bg, b_bg, loss_bg = batch_gradient_descent(X, y, lr=0.01, epochs=1000)
w_sg, b_sg, loss_sg = stochastic_gradient_descent(X, y, lr=0.01, epochs=100, batch_size=32)

print("Batch GD   w:", w_bg.round(3), "b:", round(b_bg, 3))
print("SGD        w:", w_sg.round(3), "b:", round(b_sg, 3))
print("Ground truth w:", w_true, "b:", b_true)

# ------------------------------------------------------------------
# 7. Convergence plot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(loss_bg, label="Batch GD", alpha=0.8)
plt.plot(loss_sg, label="SGD (mini-batch=32)", alpha=0.8)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 8. Finite-difference gradient check
# ------------------------------------------------------------------
def finite_diff_gradient(X, y, w, b, eps=1e-5):
    """
    Numerical gradient for verification.
    """
    dw_num = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        dw_num[i] = (mse_cost(X, y, w_plus, b) - mse_cost(X, y, w_minus, b)) / (2 * eps)

    b_plus = b + eps
    b_minus = b - eps
    db_num = (mse_cost(X, y, w, b_plus) - mse_cost(X, y, w, b_minus)) / (2 * eps)
    return dw_num, db_num

# Verify at initialization
w_test = np.zeros(n_features)
b_test = 0.0
dw_ana, db_ana = compute_gradients(X, y, w_test, b_test)
dw_num, db_num = finite_diff_gradient(X, y, w_test, b_test)

print("\nGradient check (max abs diff):")
print("dw diff:", np.max(np.abs(dw_ana - dw_num)))
print("db diff:", abs(db_ana - db_num))
```

</details>

## What You Actually Learned

- **Derivatives:** You used the derivative of the squared-error term to derive the exact update rule for each parameter.
- **Partial Derivatives:** You computed the derivative of the cost with respect to each weight independently, holding others constant. This is the core of multivariate optimization.
- **Gradient Vector:** You assembled those partial derivatives into a gradient vector and subtracted it from the parameters to minimize the loss. You saw that the gradient points in the direction of steepest ascent, and its negative is the direction of descent.
- **Batch vs. Stochastic:** You observed that batch GD gives smooth, stable convergence but is slower per step, while SGD uses noisy gradient estimates that converge faster in wall-clock time and can escape flat regions.
- **Debugging intuition:** The finite-difference check gives you a reproducible way to verify analytical gradients — a skill you will use when writing custom loss functions in production.

## Sources

- [3Blue1Brown: Gradient Descent](https://www.3blue1brown.com/lessons/gradient-descent/) — Visual intuition for how gradients guide optimization.
- [PyTorch Docs: Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — The production API you will eventually use; useful contrast with the scratch implementation.
- [Neuromatch Academy: Gradient Descent and AutoGrad](https://deeplearning.neuromatch.io/tutorials/W1D2_LinearDeepLearning/student/W1D2_Tutorial1.html) — Pedagogical walkthrough of gradient descent with PyTorch.
- [GitHub: SGD-From-Scratch](https://github.com/arsenyturin/SGD-From-Scratch) — Reference notebook illustrating SGD mechanics step by step.
- [Wikipedia: Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) — Formal definitions and convergence properties.
