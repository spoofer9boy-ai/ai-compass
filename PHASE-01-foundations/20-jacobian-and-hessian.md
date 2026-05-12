# Jacobian and Hessian

**Phase:** PHASE-01-foundations  
**Prerequisites:** 17 (Partial Derivatives), 18 (Gradient Vector)  
**Estimated Time:** 50 minutes

## Why am I learning this?

You already know how to take the derivative of a single-variable function and how to compute the gradient of a scalar-valued function of many variables. In production ML, those two ideas are not enough. A neural network layer is a *vector-valued* function: it takes a vector of inputs and returns a vector of outputs. The Jacobian matrix is the natural way to describe how every output changes with respect to every input. Without it, backpropagation through custom layers, normalizing flows, and physics-informed networks becomes guesswork.

The Hessian is the next step: it captures the *curvature* of a scalar loss landscape, not just its slope. You will rarely compute a full Hessian explicitly—modern models have billions of parameters, so the Hessian would be petabytes—but you will constantly encounter Hessian-*informed* ideas: second-order optimizers, sharpness-aware minimization, trust regions, and curvature regularization. Understanding what the Hessian *means* lets you read those papers and implement those tricks without treating them as black magic.

## Where will I be using it?

- **Backpropagation through non-scalar outputs:** When you implement a custom autograd function in PyTorch, you return a Jacobian-vector product. Knowing the Jacobian shape prevents silent shape bugs.
- **Normalizing flows and invertible networks:** RealNVP, Glow, and continuous normalizing flows require the determinant of the Jacobian to compute exact likelihoods.
- **Sharpness-Aware Minimization (SAM):** SAM perturbs weights in the direction of the Hessian's top eigenvector to find flatter minima. The method is Hessian-*inspired* even though it avoids explicit construction.
- **Second-order optimization:** Methods like L-BFGS and Hessian-free optimization use curvature information to take better steps than vanilla gradient descent.
- **Adversarial robustness:** Jacobian and Hessian regularization penalize large input sensitivities, making classifiers less fragile to adversarial perturbations.
- **Sensitivity analysis in science and engineering:** When a simulator maps parameters to observations, the Jacobian tells you which parameters matter most.

## Resources

- [PyTorch Tutorials: Jacobians, Hessians, hvp, vhp, and more](https://pytorch.org/tutorials/intermediate/jacobians_hessians.html) — Official tutorial on computing Jacobians and Hessians with `torch.autograd.functional`.
- [PyTorch Docs: torch.autograd.functional.jacobian](https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) — API reference for the Jacobian function you will actually call.
- [Wikipedia: Jacobian matrix and determinant](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) — Clean definition, geometric intuition, and the change-of-variables formula.
- [Wikipedia: Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) — Definition, second-derivative test, and connections to convexity.
- [Andrew Gibiansky: Hessian Free Optimization](https://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/) — Engineering blog on second-order methods and why explicit Hessians are usually avoided.

## Appendix

### Notation

- Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ be a vector-valued function with components $f_1, \dots, f_m$.
- The **Jacobian matrix** $\mathbf{J} \in \mathbb{R}^{m \times n}$ is defined as:

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

- When $m = 1$, the Jacobian is the transpose of the gradient: $\mathbf{J} = \nabla f^\top$.
- The **Hessian matrix** $\mathbf{H} \in \mathbb{R}^{n \times n}$ of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ is the matrix of second partial derivatives:

$$
\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

- If $f$ is twice continuously differentiable, $\mathbf{H}$ is symmetric (Schwarz's theorem).

### Common Pitfalls

- **Confusing Jacobian with gradient:** The gradient is a column vector for scalar outputs; the Jacobian is a matrix for vector outputs. In PyTorch, `torch.autograd.grad` gives a vector-Jacobian product, not the Jacobian itself.
- **Ignoring memory cost:** A full Hessian for a model with $d$ parameters has $d^2$ entries. For ResNet-50 ($d \approx 25\,\text{M}$), that is ~$6 \times 10^{14}$ values. Always use Hessian-vector products (HVP) instead.
- **Shape mismatches in custom autograd:** When writing `torch.autograd.Function`, the backward pass must return tensors whose shapes match the inputs. Drawing the Jacobian shape on paper first saves debugging time later.

### Further Reading

- [arXiv:2212.00311 — Generalizing and Improving Jacobian and Hessian Regularization](https://arxiv.org/abs/2212.00311) — Research on using Jacobian and Hessian penalties for adversarial robustness.
- [Khan Academy: Jacobian prerequisite understanding](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/jacobian/v/jacobian-prerequisite-understanding) — Video intuition for the Jacobian as a local linear approximation.
