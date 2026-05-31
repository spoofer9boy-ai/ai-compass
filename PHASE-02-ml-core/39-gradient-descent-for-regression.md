# Gradient Descent for Regression

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 18 (Gradient Vector), 37 (Linear Regression)  
**Estimated Time:** 60 minutes

## Why am I learning this?

Ordinary Least Squares gives you a exact answer: plug the data into the normal equations and you get the optimal coefficients. But that exactness comes with a hidden cost. Computing $(\mathbf{X}^\top\mathbf{X})^{-1}$ is $O(d^3)$ in the number of features, and if your design matrix has correlated columns — which it almost always does in real data — the inverse does not exist or is numerically unstable. In production, you will routinely encounter datasets with millions of samples and thousands of features. Matrix inversion at that scale is either impossibly slow or physically impossible on a single machine.

Gradient descent replaces the closed-form solution with an iterative one. Instead of solving for the minimum in one shot, you start with a guess for the weights, compute the gradient of the loss with respect to those weights, and take a small step in the opposite direction. Repeat until the loss stops improving. The per-iteration cost is $O(nd)$, linear in both samples and features, and you can stop early once the error is good enough. More importantly, gradient descent is the exact same algorithm that trains neural networks. If you understand how it works for linear regression, you understand how PyTorch's `optimizer.step()` works for a 175-billion-parameter language model. The only difference is the number of layers.

There is also a subtler reason to learn this: gradient descent forces you to confront the geometry of the loss surface. You will see how the learning rate controls step size, how the condition number of the data affects convergence speed, and why feature scaling matters. These are not abstract concerns. A learning rate that is too high will make your training loss diverge to infinity; a learning rate that is too low will waste thousands of dollars in compute. Debugging training runs is a core skill for AI engineers, and it starts here.

## Where will I be using it?

- **Large-scale linear models:** When $d > 10{,}000$ or $n > 10^6$, OLS is infeasible and gradient descent (or its stochastic variant) becomes the standard approach. Logistic regression and linear SVMs in scikit-learn default to iterative solvers for this reason.
- **Deep learning:** Every neural network is trained by gradient descent. The `torch.optim.SGD` and `torch.optim.Adam` objects you instantiate are direct descendants of the algorithm described in this file.
- **Online learning:** When data arrives in a stream and you cannot store the full design matrix, stochastic gradient descent updates the model one sample at a time. This is how ad-click prediction models at Google and Meta are updated in real time.
- **Regularized regression:** Ridge and Lasso regression add penalty terms to the loss. The normal equations become more complex or disappear entirely, but gradient descent handles the modified loss with no structural changes.
- **Optimization debugging:** Understanding convergence curves, learning rate schedules, and the relationship between gradient norm and step size is essential for training any model that does not converge instantly.

## Resources

- [Stanford CS229 Lecture Notes — Gradient Descent](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Derives batch and stochastic gradient descent for linear regression, including convergence guarantees and learning rate selection.
- [scikit-learn: SGDRegressor](https://scikit-learn.org/stable/modules/sgd.html) — The production API for large-scale linear regression with gradient descent. Covers loss functions, penalties, and convergence criteria.
- [PyTorch Docs: torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) — The optimizer you will actually use. Understand what `lr`, `momentum`, and `weight_decay` mean geometrically.
- [3Blue1Brown: Gradient descent, how neural networks learn](https://www.3blue1brown.com/lessons/gradient-descent/) — Visual intuition for why following the negative gradient minimizes a function.
- [Distill.pub: Why Momentum Really Works](https://distill.pub/2017/momentum/) — Extends basic gradient descent with momentum, the most common practical modification.

## Appendix

### Notation

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: Design matrix with $n$ samples and $d$ features.
- $\mathbf{y} \in \mathbb{R}^{n}$: Target vector.
- $\boldsymbol{\beta} \in \mathbb{R}^{d}$: Parameter vector.
- $\eta > 0$: Learning rate (step size).
- $J(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$: Mean squared error loss.

### The Update Rule

The gradient of the MSE loss with respect to $\boldsymbol{\beta}$ is:

$$
\nabla_{\boldsymbol{\beta}} J = -\frac{2}{n}\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

Batch gradient descent updates the parameters as:

$$
\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t - \eta \nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}_t)
$$

### Convergence and Learning Rate

For convex quadratic losses like MSE, there exists an optimal learning rate $\eta^* = \frac{1}{\lambda_{\max}}$ where $\lambda_{\max}$ is the largest eigenvalue of the Hessian $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$. In practice, you do not know $\lambda_{\max}$, so you choose $\eta$ via grid search or use an adaptive method like Adam.

If $\eta$ is too small, convergence is slow. If $\eta$ is too large, the iterates oscillate or diverge. A good heuristic is to start with $\eta = 0.01$ and reduce it by a factor of 10 if the loss increases.

### Common Pitfalls

- **Unscaled features:** If one feature is in dollars and another is in percentages, the loss surface becomes an elongated ellipse. Gradient descent zigzags and converges slowly. Always standardize or normalize features before training.
- **Ignoring the learning rate:** Setting $\eta = 1.0$ because "bigger steps are faster" is the most common beginner mistake. The loss will explode.
- **Confusing batch and stochastic gradient descent:** Batch GD uses the full dataset every step; SGD uses one sample. Mini-batch GD (the default in deep learning) uses a small batch size (e.g., 32–512) as a compromise.
- **Stopping too early:** If you stop at the first plateau, you may be in a flat region far from the minimum. Monitor the loss on a validation set, not just the training set.

### Further Reading

- [The Elements of Statistical Learning, Section 11.4](https://web.stanford.edu/~hastie/ElemStatLearn/) — Stochastic gradient descent and its variants in the context of large-scale learning.
- [fast.ai: A disciplined approach to neural network hyper-parameters](https://arxiv.org/abs/1803.09820) — Practical heuristics for learning rate selection and scheduling.
