# Systems of Linear Equations

**Phase:** PHASE-01-foundations  
**Prerequisites:** 04-matrix-multiplication, 06-matrix-inverse  
**Estimated Time:** 40 minutes

## Why am I learning this?

Because almost every practical problem in machine learning eventually reduces to solving a system of linear equations. Linear regression? You are solving $\mathbf{X}\boldsymbol{\beta} = \mathbf{y}$ for $\boldsymbol{\beta}$. A neural network's backward pass? You are solving for gradient updates that satisfy a system of constraints. Physics simulation, portfolio optimization, circuit analysis, computer graphics ray tracing—all of it boils down to finding the vector $\mathbf{x}$ that makes $\mathbf{A}\mathbf{x} = \mathbf{b}$ true.

You will never Gaussian-eliminate a 10,000×10,000 matrix by hand. But you will absolutely spend an afternoon wondering why `numpy.linalg.solve` is returning `LinAlgError: Matrix is singular`, or why your optimization converges to different answers depending on initialization. Understanding what a system of equations *means*—geometrically, algebraically, and numerically—is what lets you debug these failures instead of blindly swapping solvers until something works.

A system of linear equations is just asking: "What linear combination of the columns of $\mathbf{A}$ produces the vector $\mathbf{b}$?" If the answer exists, it may be unique. If it does not exist, you need to know why (inconsistent constraints). If infinitely many answers exist, you need to know what that implies for your model (underdetermined problem). These three cases—unique solution, no solution, infinitely many solutions—govern the behavior of every linear model you will ever train.

## Where will I be using it?

- **Linear regression:** The normal equations $\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$ are a linear system. When $\mathbf{X}^T\mathbf{X}$ is invertible, you get a unique solution. When it is not, regularization or pseudoinverse is required.
- **Optimization (Newton's method):** Each iteration solves $\mathbf{H}\mathbf{p} = -\mathbf{g}$ for the search direction $\mathbf{p}$, where $\mathbf{H}$ is the Hessian and $\mathbf{g}$ is the gradient. In practice this is solved with `linalg.solve`, never by explicitly inverting $\mathbf{H}$.
- **Least-squares fitting:** Overdetermined systems (more equations than unknowns) have no exact solution. You solve the normal equations to minimize the squared residual $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$.
- **Graph algorithms:** PageRank and many network flow formulations reduce to solving large sparse linear systems. Specialized iterative solvers like conjugate gradient are used instead of direct factorization.
- **Computer graphics:** Interpolating vertex positions, computing barycentric coordinates, and solving for lighting parameters all involve small linear systems.
- **Control systems:** Computing equilibrium states or optimal control inputs requires solving $\mathbf{A}\mathbf{x} = \mathbf{b}$ where $\mathbf{A}$ encodes system dynamics.

## Resources

- [MIT OpenCourseWare: Solving Ax = b (Gilbert Strang)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/ax-b-and-the-four-subspaces/solving-ax-b-row-reduced-form-r/) — The canonical lecture on elimination, pivot positions, and when systems have zero, one, or infinitely many solutions.
- [NumPy Docs: numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) — The standard API for solving dense square linear systems. Read the notes on singular matrices.
- [PyTorch Docs: torch.linalg.solve](https://pytorch.org/docs/stable/generated/torch.linalg.solve.html) — GPU-accelerated solver for batched linear systems, common in deep learning training loops.
- [SciPy Linear Algebra Tutorial](https://docs.scipy.org/doc/scipy/tutorial/linalg.html) — Covers both direct solvers (`solve`) and iterative methods for large sparse systems.
- [3Blue1Brown: Essence of Linear Algebra (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Visual intuition for what $\mathbf{A}\mathbf{x} = \mathbf{b}$ means in terms of column space and linear combinations.

## Appendix

### Notation

- $\mathbf{A}\mathbf{x} = \mathbf{b}$: A system of $m$ equations in $n$ unknowns, where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^{n}$, and $\mathbf{b} \in \mathbb{R}^{m}$.
- **Augmented matrix:** $[\mathbf{A} \mid \mathbf{b}]$ — the compact representation used in Gaussian elimination.
- **Row echelon form (REF):** An upper-triangular equivalent system obtained via row operations.
- **Reduced row echelon form (RREF):** REF with leading entries equal to 1 and zeros above and below each pivot.

### The Three Cases

For a system $\mathbf{A}\mathbf{x} = \mathbf{b}$:

1. **Unique solution:** $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A} \mid \mathbf{b}]) = n$ (number of unknowns). $\mathbf{A}$ has full column rank. If $\mathbf{A}$ is square, it is invertible and $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$.
2. **No solution:** $\text{rank}(\mathbf{A}) < \text{rank}([\mathbf{A} \mid \mathbf{b}])$. The vector $\mathbf{b}$ is not in the column space of $\mathbf{A}$. The system is **inconsistent**.
3. **Infinitely many solutions:** $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A} \mid \mathbf{b}]) < n$. There are free variables. The solution set is an affine subspace: $\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h$, where $\mathbf{x}_p$ is a particular solution and $\mathbf{x}_h$ is any vector in the null space of $\mathbf{A}$.

### Gaussian Elimination

The algorithm you learned in school, implemented in every numerical library:

1. Forward elimination: Use row operations to create zeros below each pivot.
2. Back substitution: Solve for variables starting from the last row.

Computational complexity: $O(n^3)$ for a dense $n \times n$ system. In practice, libraries use LU decomposition (factor $\mathbf{A} = \mathbf{L}\mathbf{U}$) which has the same complexity but better numerical stability and allows fast solving for multiple right-hand sides.

### When Not to Invert

If you need $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$, always prefer `solve(A, b)` over `inv(A) @ b`:

- `solve` is roughly 3× faster for a single right-hand side.
- `solve` has better numerical stability (the condition number of solving is the condition number of $\mathbf{A}$; explicit inversion squares the effective error).
- `solve` can detect and report singular or near-singular matrices more reliably.

### Common Pitfalls

- **Assuming a square system always has a solution.** A random square matrix is almost surely invertible, but real design matrices often have collinear columns (multicollinearity) or redundant features, making $\mathbf{A}^T\mathbf{A}$ singular.
- **Ignoring the shape of $\mathbf{A}$.** `numpy.linalg.solve` requires $\mathbf{A}$ to be square. For rectangular systems (overdetermined or underdetermined), use `numpy.linalg.lstsq`.
- **Confusing no solution with infinitely many solutions.** Both produce errors or warnings, but the root cause and the fix are different. No solution means your model is misspecified; infinitely many means you need regularization or constraints.
- **Solving ill-conditioned systems without checking.** If the condition number $\kappa(\mathbf{A})$ is very large, small changes in $\mathbf{b}$ cause huge changes in $\mathbf{x}$. Check `numpy.linalg.cond(A)` before trusting the output.

### Further Reading

- [Wikipedia: System of Linear Equations](https://en.wikipedia.org/wiki/System_of_linear_equations) — Comprehensive reference on theory and solution methods.
- [MIT 18.06 Course Notes](https://web.mit.edu/18.06/www/) — Gilbert Strang's complete lecture notes and problem sets.
