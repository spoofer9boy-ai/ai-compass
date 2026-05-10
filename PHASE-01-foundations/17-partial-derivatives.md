# Partial Derivatives

**Phase:** PHASE-01-foundations  
**Prerequisites:** 16 (Derivatives)  
**Estimated Time:** 45 minutes

## Why am I learning this?

In production, you will almost never write a partial derivative by hand. But you will spend hours debugging why `torch.autograd` returns a gradient of the wrong shape, or why a hyperparameter search explodes because you treated a learning-rate schedule as a constant when it should have been a variable. Partial derivatives are the lens through which every modern deep-learning framework sees the world. If you do not understand them, you are flying blind inside the autograd engine.

The single-variable derivative tells you how a function changes when you nudge its *only* input. Real models have millions of inputs—weights, biases, embeddings, attention scores, batch-normalization parameters. A partial derivative isolates the effect of nudging *one* of those inputs while holding the rest fixed. That isolation is what makes gradient descent possible: you can compute every parameter update independently, in parallel, and apply them all at once.

This file exists so that when you read PyTorch’s `grad_fn` graph or stare at a Jacobian matrix in a paper, you see a collection of simple partial derivatives rather than an intimidating wall of notation.

## Where will I be using it?

- **Autograd engines:** PyTorch, TensorFlow, and JAX all compute partial derivatives via reverse-mode automatic differentiation. Every `backward()` call is a batched partial-derivative computation.
- **Hyperparameter optimization:** When tuning a learning rate or regularization strength, you are implicitly treating the loss as a function of those hyperparameters and computing partial derivatives with respect to them.
- **Sensitivity analysis:** In finance or operations research, partial derivatives (often called *Greeks* or *elasticities*) measure how an output metric changes when one input variable shifts.
- **Physics-informed neural networks (PINNs):** Enforcing a PDE loss term requires partial derivatives of the network output with respect to spatial and temporal coordinates.
- **Jacobian and Hessian matrices:** These are simply collections of first- and second-order partial derivatives. They appear in Newton-style optimizers, trust-region methods, and curvature-aware training.

## Resources

- [Khan Academy: Partial Derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives/v/introduction-to-partial-derivatives) — Visual intuition for slicing a surface along one axis at a time.
- [PyTorch Tutorials: The Fundamentals of Autograd](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html) — How PyTorch computes partial derivatives automatically via the computation graph.
- [Wikipedia: Partial Derivative](https://en.wikipedia.org/wiki/Partial_derivative) — Formal definitions, notation variants, and higher-order extensions.
- [3Blue1Brown: Partial Differential Equations](https://www.3blue1brown.com/lessons/pdes/) — Geometric intuition for partial derivatives in the context of PDEs (relevant for PINNs and physics-ML).
- [Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) — The algorithm that chains partial derivatives to train neural networks efficiently.

## Appendix

### Notation

- $\frac{\partial f}{\partial x_i}$: The partial derivative of $f$ with respect to the variable $x_i$, holding all other variables constant.
- $\nabla f = \left[ \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right]^\top$: The gradient vector, which collects all first-order partial derivatives.
- $\frac{\partial^2 f}{\partial x_i \partial x_j}$: A mixed second-order partial derivative (differentiate first w.r.t. $x_j$, then w.r.t. $x_i$).

### Common Pitfalls

- **Forgetting which variables are held constant:** When computing $\frac{\partial f}{\partial x}$, every occurrence of $y$, $z$, etc. is treated as a constant—even if those variables are themselves functions of $x$ in a larger context. That larger context requires the *chain rule*, not a plain partial derivative.
- **Shape mismatches in autograd:** PyTorch’s `grad` output has the same shape as the input tensor w.r.t. which you differentiated. If you request gradients w.r.t. a scalar and a matrix simultaneously, the returned tuple must be unpacked carefully.
- **Confusing partial and total derivatives:** The total derivative $\frac{df}{dx}$ accounts for all paths through which $x$ influences $f$. The partial derivative $\frac{\partial f}{\partial x}$ does not. In backpropagation, the chain rule turns partial derivatives into total derivatives.

### Further Reading

- [PyTorch Docs: torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html) — Manual gradient computation for advanced use cases.
- [Distill.pub: Gradient Descent](https://distill.pub/2017/momentum/) — Interactive article on optimization, heavily reliant on partial derivatives.
