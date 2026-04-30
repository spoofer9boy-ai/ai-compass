# Matrix Transpose

**Phase:** PHASE-01-foundations  
**Prerequisites:** 01-vectors, 02-vector-operations, 03-dot-product, 04-matrix-multiplication  
**Estimated Time:** 30 minutes

## Why am I learning this?

Because matrix multiplication has strict shape rules: the inner dimensions must match. The transpose is the tool that makes those dimensions match without changing the underlying data. It is the most common shape-manipulation operation in all of machine learning.

You will transpose matrices tens of thousands of times in a single training run. Attention scores are computed as $QK^T$. Gradient updates often involve transposed weight matrices. Data loaders return $(batch, features)$ but some layers expect $(features, batch)$. Transpose fixes the shape.

## Where will I be using it?

- **Transformer attention:** Computing $QK^T$ requires transposing the key matrix so the inner dimensions align for multiplication.
- **Gradient computation:** Backpropagation through a linear layer $\mathbf{y} = \mathbf{W}\mathbf{x}$ involves $\mathbf{W}^T$.
- **Covariance matrices:** Computing $\mathbf{X}^T\mathbf{X}$ to get the covariance structure of features.
- **Data reshaping:** Switching between $(batch, seq, features)$ and $(seq, batch, features)$ for different layers.
- **Weight initialization:** Some frameworks store weights as $(in, out)$ while others use $(out, in)$; transpose bridges the two.

## Resources

- [Wikipedia: Matrix Transpose](https://en.wikipedia.org/wiki/Matrix_transpose) — Formal definition, properties, and history.
- [Math Insight: The Transpose of a Matrix](https://mathinsight.org/matrix_transpose) — Visual intuition for what transposition does geometrically.
- [PyTorch Docs: torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html) — The production API you will use daily.
- [PyTorch Docs: torch.t](https://pytorch.org/docs/stable/generated/torch.t.html) — Convenience shorthand for 2D tensors.

## Appendix

### Notation

- $\mathbf{A}^T$ or $\mathbf{A}^{\intercal}$: The transpose of matrix $\mathbf{A}$.
- If $\mathbf{A} \in \mathbb{R}^{m \times n}$, then $\mathbf{A}^T \in \mathbb{R}^{n \times m}$.
- $(\mathbf{A}^T)_{ij} = A_{ji}$.

### Key Properties

- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$ (order reverses)
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$

### Views vs. Copies

In NumPy and PyTorch, `.transpose()` and `.T` usually return a **view**, not a copy. This means zero memory overhead but also that modifying the transposed tensor can modify the original.

### Common Pitfalls

- Confusing transpose with matrix inverse. $\mathbf{A}^T \neq \mathbf{A}^{-1}$ in general.
- Forgetting that `torch.t()` only works on 2D tensors. For higher dimensions, use `torch.transpose()` or `.permute()`.
- Modifying a transposed view and accidentally corrupting the original data.
- Transposing the wrong dimensions in batched operations (e.g., $(batch, seq, dim)$ vs $(seq, batch, dim)$).

### Further Reading

- [3Blue1Brown: Dot Products and Duality](https://www.3blue1brown.com/lessons/dot-products) — Transpose appears naturally in the duality interpretation.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — See where $K^T$ appears in the attention diagram.
