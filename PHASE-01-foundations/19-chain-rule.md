# Chain Rule

**Phase:** PHASE-01-foundations  
**Prerequisites:** 16 (Derivatives), 18 (Gradient Vector)  
**Estimated Time:** 45 minutes

## Why am I learning this?

You will never write a production model without the chain rule — you just won't call it that. Every time PyTorch's `loss.backward()` runs, it is traversing your computational graph and applying the chain rule at every node. When you debug why gradients vanish in an LSTM, or why a custom autograd function returns the wrong shape, you are debugging the chain rule in disguise.

The chain rule is the mechanism that lets us compute the derivative of a function that is built from other functions. In deep learning, your network is a towering stack of nested functions: loss = f(g(h(...(x)))). The chain rule is the only reason we can efficiently ask "how does a weight in layer 1 affect the final loss?" without symbolically differentiating the entire network by hand. Without it, backpropagation — and therefore modern neural network training — does not exist.

This file exists so that when you write a custom `torch.autograd.Function` and the backward pass crashes with a shape mismatch, you understand why `grad_output` must be multiplied by the local Jacobian, and why the order of that multiplication matters.

## Where will I be using it?

- **Backpropagation:** Computing $\frac{\partial L}{\partial W}$ through every layer of a neural network by chaining local gradients.
- **Custom Autograd Functions:** Implementing `backward()` for a new PyTorch operation; the chain rule dictates what you return.
- **Gradient Checking:** Numerically verifying analytical gradients; the chain rule is the analytical ground truth.
- **Normalizing Flows & VAEs:** Change-of-variables formulas require the determinant of the Jacobian, which is built from chained transformations.
- **Optimization:** Any problem where the objective is a composition of functions (e.g., meta-learning outer loops).

## Resources

- [PyTorch Tutorials: A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — The engine that automates the chain rule for you.
- [PyTorch Blog: Overview of PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) — How the chain rule is implemented under the hood via vector-Jacobian products.
- [3Blue1Brown: Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — Visual walkthrough of the chain rule through a neural network.
- [Harvey Mudd College: Multi-Variable Chain Rule](https://math.hmc.edu/calculus/hmc-mathematics-calculus-online-tutorials/multivariable-calculus/multi-variable-chain-rule/) — Clean treatment of the multivariable case with dependency diagrams.
- [Sebastian Raschka: How to compute gradients with backpropagation](https://sebastianraschka.com/faq/docs/backprop-arbitrary.html) — Practical guide to applying the chain rule for arbitrary loss and network architectures.

## Appendix

### Notation

- $f(g(x))$: A composite function where $g$ is the inner function and $f$ is the outer function.
- $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$: The single-variable chain rule, where $u = g(x)$ and $y = f(u)$.
- $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$: The multivariable chain rule when $z = f(u, v)$, $u = g(x)$, $v = h(x)$.
- **Vector-Jacobian Product (VJP):** If $\mathbf{y} = f(\mathbf{x})$, then $\frac{\partial L}{\partial \mathbf{x}} = \left(\frac{\partial L}{\partial \mathbf{y}}\right)^T \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$. PyTorch's autograd computes this product without ever materializing the full Jacobian.

### Common Pitfalls

- **Shape mismatches in custom backward passes:** The chain rule requires matrix-multiplying the incoming gradient (VJP-style) with the local Jacobian. Getting the transpose wrong produces silently wrong gradients.
- **Confusing total and partial derivatives:** In the multivariable chain rule, every path from the output back to the target variable contributes a term. Missing a path means missing a gradient contribution.
- **Treating the chain rule as only for scalars:** In deep learning, almost every intermediate is a tensor. The "chain rule" is really a recursive application of the vector-Jacobian product.

### Further Reading

- [PyTorch Docs: Autograd mechanics](https://pytorch.org/docs/main/notes/autograd.html) — The formal rules for how PyTorch traces and differentiates computational graphs.
- [Heiner.ai: The chain rule, Jacobians, autograd, and shapes](https://heiner.ai/blog/2023/02/19/chain-rule-jacobians-autograd-shapes.html) — Deep dive into why shapes work the way they do during backprop.
