# Limits and Continuity

**Phase:** PHASE-01-foundations  
**Prerequisites:** []  
**Estimated Time:** 40 minutes

## Why am I learning this?

You will never write `lim_{x→a} f(x)` in a production model file. But you will spend hours debugging why a training loss explodes to `NaN` at a specific batch, or why a gradient descent step overshoots and oscillates wildly around a minimum. Limits and continuity are the invisible scaffolding behind every optimization algorithm you will use. They explain why gradient descent works at all: because a continuous, differentiable loss surface guarantees that small steps in parameter space produce small, predictable changes in loss. Without that guarantee, optimization becomes guesswork.

The formal definition of a limit also teaches you how to think about approximation — the core habit of numerical computing. Every float operation in PyTorch is an approximation. Understanding limits means understanding when an approximation is good enough and when it breaks down. That mindset separates engineers who tune hyperparameters blindly from engineers who reason about why a model converges or diverges.

## Where will I be using it?

- **Gradient Descent:** The entire algorithm assumes the loss function is continuous and differentiable so that gradients point in a meaningful direction.
- **Activation Functions:** ReLU is continuous but not differentiable at zero; understanding limits explains why this rarely matters in practice.
- **Numerical Stability:** Computing `softmax` requires evaluating `exp(x)`; limits tell you why subtracting the max before exponentiation prevents overflow.
- **Learning Rate Scheduling:** The idea that smaller steps should lead to smaller loss changes is a continuity argument.
- **Convergence Proofs:** Every paper that claims "our optimizer converges" is implicitly relying on limits and continuity of the objective landscape.

## Resources

- [Paul's Online Math Notes: Limits](https://tutorial.math.lamar.edu/Classes/CalcI/LimitsIntro.aspx) — Exhaustive, example-heavy calculus notes from a university professor. Free and reliable.
- [Khan Academy: Limits and Continuity](https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity) — Visual intuition with interactive exercises. Excellent if you prefer learning by doing.
- [OpenStax: Calculus Volume 3 — Limits and Continuity](https://openstax.org/books/calculus-volume-3/pages/4-2-limits-and-continuity) — Free textbook chapter extending limits to multivariable functions, which is what you actually optimize over.
- [PyTorch Docs: torch.autograd notes](https://pytorch.org/docs/stable/notes/autograd.html) — Official documentation on how PyTorch computes derivatives; the entire mechanism assumes continuity.
- [SciPy Optimization Tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) — Practical optimization in Python; every method listed assumes a continuous objective.

## Appendix

### Notation

- $\displaystyle \lim_{x \to a} f(x) = L$: As $x$ approaches $a$, the value of $f(x)$ approaches $L$.
- $\epsilon$-delta definition: For every $\epsilon > 0$, there exists a $\delta > 0$ such that if $0 < |x - a| < \delta$, then $|f(x) - L| < \epsilon$.
- $f$ is continuous at $a$ if $\displaystyle \lim_{x \to a} f(x) = f(a)$.

### Common Pitfalls

- **Confusing limit with value:** The limit at a point does not have to equal the function value at that point. Discontinuities (e.g., division by zero in loss functions) are real bugs.
- **One-sided limits matter:** In ML, you often care about limits from the right (e.g., approaching zero learning rate). The left and right limits can differ.
- **Multivariable limits are harder:** $\lim_{(x,y) \to (0,0)} f(x,y)$ must hold along every path. This is why proving continuity in high-dimensional parameter space is non-trivial.
- **Numerical limits are not analytical limits:** Floating-point arithmetic means you never actually reach the limit; you get within machine epsilon. This gap causes subtle bugs in gradient checks.

### Further Reading

- [Distill.pub: Explorable Explanations](https://distill.pub) — For visual, interactive deep dives into calculus concepts.
- [Wikipedia: Limit (mathematics)](https://en.wikipedia.org/wiki/Limit_%28mathematics%29) — Rigorous definitions and history. Verified canonical URL.
