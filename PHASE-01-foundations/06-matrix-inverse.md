# Matrix Inverse

**Phase:** PHASE-01-foundations  
**Prerequisites:** 04-matrix-multiplication, 05-matrix-transpose  
**Estimated Time:** 50 minutes

## Why am I learning this?

Because every linear transformation can be reversed—if the matrix representing it is invertible. In production, you will not hand-compute inverses. But you will spend hours debugging why `numpy.linalg.inv` throws a `LinAlgError: Singular matrix`, or why your linear regression coefficients explode to infinity. Understanding when a matrix has an inverse, what it means geometrically, and why numerical libraries sometimes refuse to compute it is the difference between staring at an error and fixing it in two minutes.

The inverse is the algebraic undo button. If matrix multiplication by $\mathbf{A}$ pushes a vector in some direction, multiplying by $\mathbf{A}^{-1}$ pulls it back. In machine learning, this appears whenever you need to solve for unknowns: finding optimal weights in linear regression, updating beliefs in Kalman filters, or even inverting covariance matrices for Gaussian processes. When the inverse does not exist—and it often does not—you need to know why, and what to do instead.

## Where will I be using it?

- **Linear regression (Normal Equations):** Solving $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ for the least-squares solution. When $\mathbf{X}^T\mathbf{X}$ is singular, regularization (ridge regression) is the fix.
- **Optimization:** Newton's method uses the inverse Hessian to choose step directions. In practice, no one inverts the full Hessian directly; they solve linear systems instead. But the conceptual framework is the inverse.
- **Kalman filters:** The update step inverts the innovation covariance to compute the Kalman gain. This is why numerical stability matters in state estimation.
- **Gaussian processes:** Computing the predictive distribution requires $(\mathbf{K} + \sigma^2\mathbf{I})^{-1}$, where $\mathbf{K}$ is the kernel matrix. Cholesky decomposition is preferred for numerical stability, but the math is still the inverse.
- **Computer graphics:** Undoing a camera transform requires the inverse of the view matrix. Game engines store both forward and inverse transforms for efficiency.
- **Control theory:** Inverting the plant model to design a controller. If the model is not invertible, the system is non-minimum phase and requires approximation.

## Resources

- [MIT OpenCourseWare: Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Lectures 2 and 3 cover the geometry of inverses and elimination. The canonical academic source.
- [3Blue1Brown: Linear Algebra](https://www.3blue1brown.com/?topic=linear-algebra) — Visual intuition for what matrix inversion does geometrically (undoing a linear transformation).
- [NumPy Docs: numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) — The production API for computing inverses, with notes on singular matrices.
- [PyTorch Docs: torch.linalg.inv](https://pytorch.org/docs/stable/generated/torch.linalg.inv.html) — GPU-accelerated matrix inversion for batched tensors.
- [Wikipedia: Invertible Matrix](https://en.wikipedia.org/wiki/Invertible_matrix) — Exhaustive reference on equivalent conditions for invertibility and computational methods.

## Appendix

### Notation

- $\mathbf{A}^{-1}$: The inverse of square matrix $\mathbf{A}$.
- $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$.
- If $\mathbf{A} \in \mathbb{R}^{n \times n}$, then $\mathbf{A}^{-1} \in \mathbb{R}^{n \times n}$ (only square matrices can have inverses).

### Conditions for Invertibility

A square matrix $\mathbf{A}$ is invertible if and only if any of the following hold:

- Its determinant is non-zero: $\det(\mathbf{A}) \neq 0$.
- Its rows (and columns) are linearly independent.
- The equation $\mathbf{A}\mathbf{x} = \mathbf{0}$ has only the trivial solution $\mathbf{x} = \mathbf{0}$.
- It has full rank: $\text{rank}(\mathbf{A}) = n$.
- All eigenvalues are non-zero.

If any of these fail, $\mathbf{A}$ is **singular** and has no inverse.

### Computing the Inverse

- **Gauss-Jordan elimination:** Augment $\mathbf{A}$ with $\mathbf{I}$ and row-reduce until the left side becomes $\mathbf{I}$. The right side is $\mathbf{A}^{-1}$. This is $O(n^3)$.
- **LU decomposition:** Factor $\mathbf{A} = \mathbf{L}\mathbf{U}$, then solve for the inverse by forward/back substitution. This is what NumPy and LAPACK do under the hood.
- **Formula for 2×2:** If $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, then $\mathbf{A}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$, provided $ad-bc \neq 0$.

### The Moore-Penrose Pseudoinverse

When $\mathbf{A}$ is not square, or is square but singular, the inverse does not exist. The **pseudoinverse** $\mathbf{A}^+$ generalizes the concept and gives the minimum-norm least-squares solution. It is computed via the SVD: $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$. In NumPy, use `numpy.linalg.pinv`.

### Common Pitfalls

- **Assuming all square matrices are invertible.** A random square matrix is almost surely invertible, but real data matrices often are not due to collinearity or missing data.
- **Computing the inverse explicitly when you only need to solve a linear system.** If you need $\mathbf{A}^{-1}\mathbf{b}$, use `numpy.linalg.solve(A, b)` instead of `inv(A) @ b`. It is faster and more numerically stable.
- **Inverting large matrices without checking conditioning.** The condition number $\kappa(\mathbf{A}) = \|\mathbf{A}\|\|\mathbf{A}^{-1}\|$ measures sensitivity to errors. If $\kappa(\mathbf{A})$ is huge (e.g., $>10^{12}$), the inverse is unreliable in floating-point arithmetic.
- **Confusing the inverse with the transpose.** $\mathbf{A}^{-1} \neq \mathbf{A}^T$ in general. They are equal only for **orthogonal matrices** (rotations and reflections).

### Further Reading

- [Wolfram MathWorld: Matrix Inverse](https://mathworld.wolfram.com/MatrixInverse.html) — Properties and formulas.
- [arXiv: The Pseudoinverse of A=CR is A+=R+C+](https://arxiv.org/abs/2305.01716) — Gilbert Strang on when and why the pseudoinverse breaks down.
