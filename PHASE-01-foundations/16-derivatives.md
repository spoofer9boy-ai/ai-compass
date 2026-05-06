# Derivatives

**Phase:** PHASE-01-foundations  
**Prerequisites:** [15 - Limits and Continuity]  
**Estimated Time:** 50 minutes

## Why am I learning this?

You will never compute a derivative by hand in production. PyTorch, TensorFlow, and JAX all have `autograd` engines that handle differentiation automatically. But you will spend hours debugging why your loss isn't decreasing, why gradients are exploding to infinity, or why certain weights aren't updating at all. When that happens, you need to know what a derivative actually represents—not just how to call `.backward()`.

Derivatives are the mathematical engine behind every neural network training loop. They tell you which direction to adjust each parameter to reduce loss. Without understanding derivatives, backpropagation is just magic incantations. With understanding, you can diagnose vanishing gradients in RNNs, spot when activation functions are saturating, and reason about why learning rates matter. This file exists so that when you see `grad_fn` in a PyTorch tensor, you know exactly what computation graph led there.

The derivative is also your first step into optimization theory. Gradient descent, Adam, and second-order methods all assume you can compute (or approximate) how a function changes with respect to its inputs. If you don't understand what that means mathematically, you can't reason about convergence, local minima, or why momentum helps.

## Where will I be using it?

- **Neural Network Training:** Computing gradients of loss with respect to weights during backpropagation.
- **PyTorch Autograd:** Understanding what happens when you call `.backward()` on a scalar loss.
- **Learning Rate Scheduling:** Reasoning about how step size relates to the magnitude of derivatives.
- **Activation Function Design:** Understanding why ReLU avoids vanishing gradients better than sigmoid.
- **Optimization Algorithms:** Implementing gradient descent, momentum, and adaptive methods.
- **Sensitivity Analysis:** Measuring how model outputs change with respect to input perturbations.
- **Physics-Informed Neural Networks:** Encoding differential equations as loss terms.

## Resources

- [3Blue1Brown: Backpropagation Calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus/) — Visual intuition for how derivatives flow through computation graphs.
- [PyTorch Tutorials: Automatic Differentiation](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — The `torch.autograd` engine you'll use daily.
- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — How derivatives enable optimization-based feature visualization.
- [arXiv: Gradients without Backpropagation](https://arxiv.org/abs/2202.08587) — Alternative approaches to computing derivatives when backprop is impractical.
- [Khan Academy: Derivative Definition](https://www.khanacademy.org/math/calculus-1/derivatives-intro) — Rigorous foundation for the limit definition.

## Appendix

### Notation

- $f'(x)$ or $\frac{df}{dx}$: The derivative of $f$ with respect to $x$, representing the instantaneous rate of change.
- $\frac{\partial f}{\partial x}$: Partial derivative (used when $f$ depends on multiple variables).
- $\nabla f$: The gradient vector, containing all partial derivatives of a multivariate function.
- $\dot{x}$: Newton's notation for time derivative (common in physics).

### Common Derivatives

| Function $f(x)$ | Derivative $f'(x)$ |
|-----------------|-------------------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| $\tanh(x)$ | $1 - \tanh^2(x)$ |
| $\text{ReLU}(x) = \max(0, x)$ | $1$ if $x > 0$, else $0$ (undefined at 0) |

### Rules of Differentiation

**Sum Rule:** $(f + g)' = f' + g'$

**Product Rule:** $(fg)' = f'g + fg'$

**Quotient Rule:** $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$

**Chain Rule:** $(f \circ g)' = (f' \circ g) \cdot g'$ — The foundation of backpropagation.

### Connection to Limits

The derivative is defined as a limit:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

This represents the slope of the tangent line at point $x$, obtained by taking the limit of secant slopes as the interval $h$ shrinks to zero.

### Common Pitfalls

- **Treating derivatives as fractions:** $\frac{dy}{dx}$ is not actually a fraction (though it often behaves like one in chain rule calculations).
- **Ignoring non-differentiable points:** ReLU is not differentiable at $x=0$; implementations typically return 0 or 1 arbitrarily.
- **Confusing derivative with gradient:** The derivative is a scalar for scalar functions; the gradient is a vector for multivariate functions.
- **Forgetting the chain rule:** In deep networks, you must propagate gradients through every operation, not just the final layer.

### Further Reading

- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — Deep dive into how PyTorch builds and traverses computation graphs.
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) — Stanford course notes on computing gradients in neural networks.
- [Wolfram MathWorld: Derivative](https://mathworld.wolfram.com/Derivative.html) — Comprehensive reference for mathematical properties.
