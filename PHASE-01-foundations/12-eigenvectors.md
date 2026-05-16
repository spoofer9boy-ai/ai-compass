# Eigenvectors

**Phase:** PHASE-01-foundations  
**Prerequisites:** 11 (Eigenvalues)  
**Estimated Time:** 50 minutes

## Why am I learning this?

Eigenvalues tell you *how much* a matrix scales space. Eigenvectors tell you *in what direction* that scaling happens. In production ML, you will rarely compute them by hand, but you will spend hours debugging why PCA returned a component that looks like noise, or why your spectral clustering assignment is unstable. Understanding eigenvectors means you can look at a matrix and predict which directions matter — and which are just numerical artifacts.

In neural networks, eigenvectors of the Hessian reveal which parameter directions are most sensitive to change. In graph ML, the eigenvectors of a graph Laplacian encode community structure. In computer vision, the eigenvectors of a covariance matrix *are* the principal components that compress an image without losing perceptual quality. If eigenvalues are the "what," eigenvectors are the "where." You need both.

## Where will I be using it?

- **Principal Component Analysis (PCA):** The eigenvectors of the covariance matrix are the principal components — the orthogonal directions of maximum variance in your data.
- **Spectral Clustering:** Eigenvectors of the graph Laplacian reveal cluster structure; k-means is run on these eigenvectors, not the raw data.
- **PageRank / Graph Algorithms:** The dominant eigenvector of the web-link adjacency matrix (with damping) is the PageRank score vector.
- **Stability Analysis in Neural Networks:** Eigenvectors of the Hessian identify flat vs. sharp minima; sharp minima (large eigenvalues in some directions) generalize worse.
- **Physics Simulation / Robotics:** Eigenvectors of inertia tensors and stiffness matrices describe the natural modes of vibration or deformation.
- **Recommender Systems (SVD-based):** The left and right singular vectors are eigenvectors of $AA^T$ and $A^TA$; they form the latent factor embeddings.

## Resources

- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues/) — Visual intuition for why eigenvectors stay on their own span during a transformation.
- [MIT 18.06: Eigenvalues and Eigenvectors (PDF)](https://math.mit.edu/~gs/linearalgebra/ila6/ila6_6_1.pdf) — Gilbert Strang's chapter on the geometry and algebra of eigenvectors.
- [Wikipedia: Eigenvalues and Eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) — Rigorous definitions, properties, and the full mathematical picture.
- [Spectral Clustering Explained](https://towardsdatascience.com/spectral-clustering-explained-how-eigenvectors-reveal-complex-cluster-structures/) — How eigenvectors of the graph Laplacian expose cluster boundaries. **Note:** Medium may block automated requests; if the link is inaccessible, use the [Wikipedia: Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering) or [scikit-learn Spectral Clustering docs](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) instead.
- [PCA and Eigenvectors](https://www.datasciencebase.com/intermediate/linear-algebra/principal-component-analysis/) — Why eigenvectors of the covariance matrix are the principal components. **Note:** This site may return 403; if blocked, use the Wikipedia PCA article or scikit-learn PCA docs instead.

## Appendix

### Notation

- $\mathbf{A} \in \mathbb{R}^{n \times n}$: A square matrix.
- $\lambda$: An eigenvalue (scalar) of $\mathbf{A}$.
- $\mathbf{v} \in \mathbb{R}^n$: An eigenvector of $\mathbf{A}$ corresponding to $\lambda$, satisfying $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$.
- $E_\lambda = \{\mathbf{v} \mid \mathbf{A}\mathbf{v} = \lambda\mathbf{v}\}$: The eigenspace for eigenvalue $\lambda$ (includes the zero vector; eigenvectors are the non-zero members).

### Computing Eigenvectors (Conceptual)

Given an eigenvalue $\lambda$, find the eigenvectors by solving the homogeneous system:

$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

The solution space is the null space (kernel) of $(\mathbf{A} - \lambda\mathbf{I})$. Any non-zero vector in that null space is an eigenvector for $\lambda$.

**Example:**
For $\mathbf{A} = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$, eigenvalues are $\lambda_1 = 5$ and $\lambda_2 = 2$.

For $\lambda_1 = 5$:
$$(\mathbf{A} - 5\mathbf{I}) = \begin{bmatrix} -1 & 2 \\ 1 & -2 \end{bmatrix}$$
Solving gives $v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ (or any scalar multiple).

For $\lambda_2 = 2$:
$$(\mathbf{A} - 2\mathbf{I}) = \begin{bmatrix} 2 & 2 \\ 1 & 1 \end{bmatrix}$$
Solving gives $v_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ (or any scalar multiple).

### Key Properties

1. **Scaling Invariance:** If $\mathbf{v}$ is an eigenvector, so is $c\mathbf{v}$ for any non-zero scalar $c$. Eigenvectors are directions, not specific vectors. In practice, they are normalized to unit length.
2. **Linear Independence:** Eigenvectors corresponding to *distinct* eigenvalues are linearly independent. If an $n \times n$ matrix has $n$ distinct eigenvalues, its eigenvectors form a basis for $\mathbb{R}^n$.
3. **Orthogonality (Symmetric Matrices):** For a real symmetric matrix ($\mathbf{A} = \mathbf{A}^T$), eigenvectors corresponding to different eigenvalues are orthogonal. This is why PCA components are orthogonal.
4. **Defective Matrices:** Some matrices do not have a full set of linearly independent eigenvectors. These are called *defective* or *non-diagonalizable* matrices. In ML, you usually work with symmetric covariance matrices or use SVD as a more robust alternative.
5. **Dominant Eigenvector:** The eigenvector associated with the largest-magnitude eigenvalue is called the *dominant* eigenvector. Power iteration converges to this vector and is the basis for PageRank.

### Common Pitfalls

- **Forgetting normalization:** Libraries return normalized eigenvectors, but manual computation may not. Always check if $\|\mathbf{v}\| = 1$ when comparing.
- **Confusing eigenvectors of $\mathbf{A}$ with $\mathbf{A}^T$:** They are generally different. Left eigenvectors satisfy $\mathbf{v}^T\mathbf{A} = \lambda\mathbf{v}^T$.
- **Assuming real eigenvectors:** Non-symmetric real matrices can have complex eigenvalues and eigenvectors. In ML, covariance matrices are symmetric, so this is usually not an issue for PCA.
- **Sign ambiguity:** If $\mathbf{v}$ is an eigenvector, so is $-\mathbf{v}$. PCA components may flip sign between different library versions or random seeds — this is mathematically valid and does not affect interpretation.
- **Eigenspace dimension > 1:** When eigenvalues repeat (algebraic multiplicity > 1), the eigenspace may have multiple independent eigenvectors. PCA on perfectly spherical data, for example, has degenerate eigenvalues.

### Further Reading

- [Gilbert Strang: Introduction to Linear Algebra (6th Ed.)](https://math.mit.edu/~gs/linearalgebra/) — The definitive textbook treatment of eigenvectors, eigenspaces, and diagonalization.
- [Distill.pub: Explorable Explanations](https://distill.pub) — For interactive visualizations of matrix decompositions.
- [NumPy Docs: numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — The API you will actually use to compute eigenvectors.
