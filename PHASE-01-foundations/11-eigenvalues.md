# Eigenvalues

**Phase:** PHASE-01-foundations  
**Prerequisites:** 4 (Matrix Multiplication), 7 (Determinants)  
**Estimated Time:** 55 minutes

## Why am I learning this?

Eigenvalues are the numbers that tell you how much a linear transformation stretches or shrinks things along its natural axes. In production ML, you will never hand-calculate an eigenvalue. But you will spend hours debugging why PCA whitening collapsed your feature variances, or why your spectral clustering implementation converged to garbage. When that happens, you need to know that an eigenvalue measures the scaling factor along an eigenvector — and that a zero eigenvalue means the transformation collapses a dimension. That intuition turns a black-box NumPy call into a debuggable tool.

The concept also appears in stability analysis: in dynamical systems and recurrent networks, eigenvalues of the transition matrix determine whether signals explode, decay, or oscillate. An eigenvalue with magnitude greater than one means the system is unstable. In neural ODEs and RNN training, that intuition is the difference between a model that trains and one that diverges.

Finally, eigenvalues are the gateway to almost every dimensionality-reduction and spectral method in machine learning. PCA, LDA, spectral clustering, graph Laplacians, and PageRank all reduce to finding eigenvalues of some matrix. If you skip this, you are memorizing APIs instead of understanding why they work.

## Where will I be using it?

- **Principal Component Analysis (PCA):** The eigenvalues of the covariance matrix tell you how much variance each principal component captures. You keep the components with the largest eigenvalues.
- **Spectral Clustering:** The eigenvalues of the graph Laplacian determine the number and quality of clusters. The multiplicity of the zero eigenvalue equals the number of connected components.
- **PageRank / Random Walks:** The stationary distribution of a Markov chain is the eigenvector associated with eigenvalue 1 of the transition matrix.
- **Stability of Dynamical Systems:** In RNNs and neural ODEs, eigenvalues of the Jacobian at fixed points determine local stability.
- **Matrix Condition Number:** The ratio of largest to smallest eigenvalue (in magnitude) of a matrix tells you how numerically unstable inversion or solving linear systems will be.
- **Quadratic Forms and Optimization:** The definiteness of a Hessian matrix (positive/negative definite) is determined by the signs of its eigenvalues, which tells you whether a critical point is a minimum, maximum, or saddle.

## Resources

- [3Blue1Brown: Eigenvalues and Eigenvectors](https://www.3blue1brown.com/lessons/eigenvalues) — Visual intuition for what eigenvalues actually measure.
- [NumPy Docs: numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — The API you will actually use in Python.
- [PyTorch Docs: torch.linalg.eig](https://pytorch.org/docs/stable/generated/torch.linalg.eig.html) — GPU-accelerated eigenvalue computation for deep learning workflows.
- [Wikipedia: Eigenvalues and Eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) — Comprehensive reference for definitions, properties, and computational methods.
- [Wolfram MathWorld: Eigenvalue](https://mathworld.wolfram.com/Eigenvalue.html) — Mathematical properties, formulas, and special cases.

## Appendix

### Notation

- $\lambda$: An eigenvalue of a square matrix $\mathbf{A}$.
- $\mathbf{A} \in \mathbb{R}^{n \times n}$: A square matrix.
- $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$: The characteristic equation whose roots are the eigenvalues.
- $\mathbf{I}$: The identity matrix of the same dimension as $\mathbf{A}$.

### How to compute eigenvalues (conceptually)

1. Form the matrix $(\mathbf{A} - \lambda \mathbf{I})$.
2. Compute its determinant: $\det(\mathbf{A} - \lambda \mathbf{I})$.
3. This gives a polynomial in $\lambda$ of degree $n$ (the characteristic polynomial).
4. Find the roots of that polynomial. Those roots are the eigenvalues.

In practice, nobody does this for $n > 3$. Libraries use iterative methods like the QR algorithm or divide-and-conquer. But the characteristic equation is still the theoretical foundation.

### Key properties

- A matrix has exactly $n$ eigenvalues in $\mathbb{C}$ counting multiplicities (Fundamental Theorem of Algebra).
- The product of all eigenvalues equals $\det(\mathbf{A})$.
- The sum of all eigenvalues equals $\text{tr}(\mathbf{A})$ (the trace, or sum of diagonal entries).
- If $\mathbf{A}$ is symmetric, all eigenvalues are real.
- If $\mathbf{A}$ is positive definite, all eigenvalues are positive.
- A matrix is singular (non-invertible) if and only if it has at least one zero eigenvalue.

### Common Pitfalls

- Confusing algebraic multiplicity (how many times an eigenvalue appears as a root) with geometric multiplicity (how many linearly independent eigenvectors correspond to it). They are not always equal.
- Assuming all matrices have a full set of eigenvectors. Defective matrices do not.
- Expecting eigenvalues of non-symmetric real matrices to be real. They can be complex.
- Using `numpy.linalg.eig` on a symmetric matrix when `numpy.linalg.eigh` (optimized for Hermitian/symmetric matrices) is faster and guarantees real eigenvalues.

### Further Reading

- [Gilbert Strang: Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) — Chapters on eigenvalues and their role in PCA and graphs.
- [Distill.pub: Principal Component Analysis](https://distill.pub/) — Interactive visualizations of how eigenvalues drive dimensionality reduction.
