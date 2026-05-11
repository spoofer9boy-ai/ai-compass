# Gradient Vector

**Phase:** PHASE-01-foundations  
**Prerequisites:** 17 (Partial Derivatives)  
**Estimated Time:** 40 minutes

## Why am I learning this?

You already know how to take a partial derivative: hold every variable constant except one, then differentiate. In isolation, partial derivatives are just slopes along individual axes. But in machine learning, you almost never care about a single slope—you care about which direction to move *all* of your parameters at once to make the loss go down. That direction is the **gradient vector**.

You will never write a paper about the gradient. But you will spend weeks of your career staring at training curves, tuning learning rates, and wondering why your model converges slowly or diverges. Every one of those moments traces back to the gradient: what it points to, how large it is, and whether you are following it intelligently. This file exists so that `optimizer.step()` is not a black box.

## Where will I be using it?

- **Training neural networks:** The gradient of the loss with respect to all weights and biases tells PyTorch/TensorFlow exactly how to update them.
- **Feature importance:** In linear models, the gradient of the prediction with respect to an input feature tells you how sensitive the output is to that feature.
- **Adversarial examples:** Small input perturbations in the direction of the gradient can flip model predictions—this is how FGSM attacks work.
- **Physics-informed ML:** Gradients of a PDE residual with respect to network inputs enforce physical constraints.
- **Optimization debugging:** Gradient norms reveal vanishing or exploding gradients before they crash your training run.

## Resources

- [3Blue1Brown: Gradient descent, how neural networks learn](https://www.3blue1brown.com/lessons/gradient-descent/) — Visual intuition for why the gradient is the direction of steepest ascent.
- [PyTorch Tutorials: A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — How PyTorch computes and stores gradients automatically.
- [BetterExplained: Vector Calculus — Understanding the Gradient](https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/) — Intuitive explanation of the gradient as a vector of partial derivatives.
- [Math Insight: The gradient vector](https://mathinsight.org/gradient_vector) — Formal definition and geometric interpretation.
- [Wikipedia: Gradient](https://en.wikipedia.org/wiki/Gradient) — Comprehensive reference on properties, notation, and generalizations.

## Appendix

### Notation

For a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$ with inputs $x_1, x_2, \dots, x_n$:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

The gradient $\nabla f(\mathbf{x})$ is a vector in the same space as $\mathbf{x}$. It points in the direction of the steepest increase of $f$ at $\mathbf{x}$, and its magnitude $\|\nabla f(\mathbf{x})\|$ equals the rate of increase in that direction.

### Directional Derivative Connection

The directional derivative of $f$ in the direction of a unit vector $\mathbf{u}$ is:

$$
D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \|\nabla f(\mathbf{x})\| \cos\theta
$$

This is maximized when $\mathbf{u}$ aligns with $\nabla f(\mathbf{x})$ ($\cos\theta = 1$), confirming that the gradient points in the direction of steepest ascent. The negative gradient $-\nabla f(\mathbf{x})$ therefore points in the direction of steepest descent—exactly what gradient descent exploits.

### Common Pitfalls

- **Forgetting the gradient is a vector, not a scalar.** Each component is a partial derivative; the whole thing lives in input space, not output space.
- **Confusing gradient with Jacobian.** The gradient is for scalar outputs; the Jacobian generalizes it to vector-valued functions.
- **Ignoring magnitude.** A large gradient means the function is changing rapidly; a near-zero gradient means you are near a critical point (minimum, maximum, or saddle).
- **Assuming the gradient always points toward the global minimum.** It only points toward the nearest local minimum along the steepest path. Non-convex landscapes (like neural network loss surfaces) are full of saddle points and local minima.

### Further Reading

- [Distill.pub: Why Momentum Really Works](https://distill.pub/2017/momentum/) — How gradient descent is modified in practice with momentum.
- [PyTorch Docs: torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html) — Manual gradient computation for advanced use cases.
