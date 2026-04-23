# Matrix Multiplication

**Phase:** PHASE-01-foundations  
**Prerequisites:** 01-vectors, 02-vector-operations, 03-dot-product  
**Estimated Time:** 60 minutes

## Why am I learning this?

Because matrix multiplication is how you do billions of dot products at once. A single matrix multiply can replace nested loops that would take minutes to run. Every layer of every neural network is a matrix multiplication followed by a nonlinearity.

If you understand matrix multiplication, you understand why GPUs exist. It is the most important operation in all of modern AI.

## Where will I be using it?

- **Neural network layers:** $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ is a matrix-vector multiplication. Stacking layers means stacking matrix multiplications.
- **Attention in transformers:** The entire attention mechanism is a sequence of matrix multiplications: $QK^T$, then softmax, then multiply by $V$.
- **Batch processing:** Processing a batch of 1,000 examples in parallel is a matrix-matrix multiplication.
- **Recommender systems:** Computing all user-item scores at once is a matrix multiplication of the user embedding matrix and the item embedding matrix.
- **Computer graphics:** Converting 3D world coordinates to 2D screen coordinates uses a projection matrix multiplied by vertex position vectors.
- **State transitions:** In Markov chains and some RL algorithms, transition probabilities are stored as matrices.

## Resources

- [3Blue1Brown: Matrix Multiplication as Composition](https://www.3blue1brown.com/lessons/matrix-multiplication) — The composition intuition.
- [Khan Academy: Matrix Multiplication](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/a/multiplying-matrices) — Step-by-step mechanics.
- [PyTorch Docs: torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html) — The function you will call thousands of times.
- [NVIDIA Blog: What Is a Tensor Core?](https://developer.nvidia.com/blog/tensor-core-ai-performance-milestones/) — Why matrix multiplication drives hardware design.

## Appendix

### Notation

- $\mathbf{C} = \mathbf{A}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{m 	imes n}$, $\mathbf{B} \in \mathbb{R}^{n 	imes p}$, and $\mathbf{C} \in \mathbb{R}^{m 	imes p}$.
- $C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$ (dot product of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$).

### Shapes and Dimensions

The inner dimensions must match: $(m 	imes oldsymbol{n})(oldsymbol{n} 	imes p) = (m 	imes p)$.

### Matrix-Vector Multiplication

A special case where $\mathbf{B}$ is a column vector ($n 	imes 1$), yielding a vector ($m 	imes 1$). This is one neuron layer.

### Batch Matrix Multiplication

PyTorch `torch.bmm` handles batches of matrices: `(b, m, n) × (b, n, p) → (b, m, p)`. This is how transformers process entire batches of sequences.

### Common Pitfalls

- Dimension mismatch: the most common PyTorch error is `mat1 and mat2 shapes cannot be multiplied`.
- Confusing matrix multiplication with element-wise multiplication (`*` in NumPy).
- Forgetting that matrix multiplication is not commutative: $\mathbf{AB} 
eq \mathbf{BA}$ in general.
- Not leveraging broadcasting rules for batch operations.

### Further Reading

- [Blog: An Intuitive Introduction to Matrix Multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html) — Gentle, visual introduction.
- [Paper: FlashAttention-2](https://arxiv.org/abs/2307.08691) — How to make attention's matrix multiplications memory-efficient.
