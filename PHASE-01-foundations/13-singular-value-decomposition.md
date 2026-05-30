# Singular Value Decomposition

**Phase:** PHASE-01-foundations
**Prerequisites:** 4 (Matrix Multiplication), 11 (Eigenvalues), 12 (Eigenvectors)
**Estimated Time:** 60 minutes

## Why am I learning this?

You have already learned that eigendecomposition breaks a square matrix into eigenvalues and eigenvectors. That is powerful, but it has a hard limitation: it only works on square matrices, and even then only on matrices that are diagonalizable. In production machine learning, the matrices you deal with are almost never square and almost never perfectly behaved. A user–item interaction matrix in a recommender system is rectangular. A batch of image embeddings is rectangular. A term–document matrix in NLP is rectangular. Eigendecomposition cannot touch these directly.

Singular Value Decomposition (SVD) is the generalization that removes every one of those restrictions. It works on any real or complex matrix of any shape. It is numerically stable. It is the algorithm that powers latent semantic analysis, collaborative filtering, image compression, and the low-rank approximations that make large models tractable. If you ever use `numpy.linalg.svd`, `torch.svd`, or `sklearn.decomposition.TruncatedSVD`, you are leaning on fifty years of numerical analysis that started with this exact factorization. Understanding what the three output matrices mean—and why the singular values are sorted—lets you debug dimension mismatches, choose truncation ranks, and explain why your compressed embedding still preserves 95 % of the variance.

## Where will I be using it?

- **Dimensionality Reduction (PCA):** PCA is SVD on a centered data matrix. The singular values tell you how much variance each principal component captures. You will call `sklearn.decomposition.PCA` or `TruncatedSVD` and need to decide how many components to keep.
- **Recommender Systems:** Matrix factorization methods (the Netflix Prize winners) factorize a user–item rating matrix into two lower-rank matrices. That factorization is a truncated SVD in disguise.
- **Image & Signal Compression:** Keep the top *k* singular values and drop the rest. A grayscale image becomes a low-rank approximation that stores only `k*(m+n+1)` numbers instead of `m*n`.
- **Noise Reduction:** Small singular values often correspond to noise. Zeroing them out before reconstruction is a standard preprocessing step in spectroscopy and sensor data.
- **Pseudoinverse & Least Squares:** `numpy.linalg.pinv` uses SVD to compute the Moore–Penrose pseudoinverse, which solves least-squares problems even when the matrix is rank-deficient or non-square.
- **Latent Semantic Analysis (LSA):** In NLP, a term–document matrix is decomposed via truncated SVD to uncover hidden topic structure. This predates modern transformers and is still used in information retrieval baselines.

## Resources

- [Wikipedia: Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) — Comprehensive mathematical treatment, history, and connections to eigendecomposition.
- [NumPy Docs: numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) — The API you will actually call. Explains the `full_matrices`, `compute_uv`, and `hermitian` flags.
- [3Blue1Brown: What is the Singular Value Decomposition?](https://www.youtube.com/watch?v=CpD9XlTu3ys) — Visual, geometric intuition for how SVD rotates, scales, and rotates again.
- [Dennis Miczek: SVD Image Compression, Explained](https://dmicz.github.io/machine-learning/svd-image-compression/) — Hands-on walkthrough of low-rank approximation with concrete Python code and compression ratios.
- [Wikipedia: Matrix Factorization (Recommender Systems)](https://en.wikipedia.org/wiki/Matrix_factorization_%28recommender_systems%29) — How truncated SVD and its variants (SVD++, Funk SVD) power collaborative filtering at scale.

## Appendix

### Notation

- $\mathbf{A} \in \mathbb{R}^{m \times n}$: an arbitrary real matrix with $m$ rows and $n$ columns.
- $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$: the SVD of $\mathbf{A}$.
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: orthogonal matrix of left-singular vectors.
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: rectangular diagonal matrix with non-negative singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$ on the diagonal, where $r = \text{rank}(\mathbf{A})$.
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: orthogonal matrix of right-singular vectors.
- $\mathbf{V}^T$: transpose of $\mathbf{V}$ (conjugate transpose if complex).

### Compact (Truncated) SVD

In practice you rarely need the full $\mathbf{U}$ and $\mathbf{V}$. If $\text{rank}(\mathbf{A}) = r$, you can write:

$$
\mathbf{A} = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

where $\mathbf{u}_i$ and $\mathbf{v}_i$ are the $i$-th columns of $\mathbf{U}$ and $\mathbf{V}$. A rank-$k$ approximation keeps only the first $k$ terms:

$$
\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

This is the Eckart–Young–Mirsky theorem: $\mathbf{A}_k$ is the best rank-$k$ approximation of $\mathbf{A}$ in both Frobenius and spectral norm.

### Relationship to Eigendecomposition

- The singular values of $\mathbf{A}$ are the square roots of the eigenvalues of $\mathbf{A}^T\mathbf{A}$ (or $\mathbf{A}\mathbf{A}^T$).
- The columns of $\mathbf{V}$ are the eigenvectors of $\mathbf{A}^T\mathbf{A}$.
- The columns of $\mathbf{U}$ are the eigenvectors of $\mathbf{A}\mathbf{A}^T$.

If $\mathbf{A}$ is square, symmetric, and positive semi-definite, its SVD coincides with its eigendecomposition.

### Common Pitfalls

- **Confusing shapes:** `numpy.linalg.svd` returns `U, s, Vh` where `Vh` is already $\mathbf{V}^T$. Do not transpose it again.
- **Full vs. reduced:** By default `numpy.linalg.svd` returns full $\mathbf{U}$ and $\mathbf{V}^T$ ($m \times m$ and $n \times n$). For large matrices, use `full_matrices=False` to get the compact shapes ($m \times r$ and $r \times n$).
- **Forgetting to center for PCA:** PCA is SVD on the *centered* data matrix. If you skip mean subtraction, your first component captures the mean, not the direction of maximum variance.
- **Choosing *k* by eye:** Use the cumulative explained variance ratio $\sum_{i=1}^{k} \sigma_i^2 / \sum_{i=1}^{r} \sigma_i^2$ to pick $k$ with a quantitative threshold (e.g., 95 %).

### Further Reading

- [Trokas AI Primer: SVD](https://trokas.github.io/ai_primer/SVD.html) — Interactive geometric illustrations and Python snippets.
- [ArXiv: Image Compression Using Singular Value Decomposition](https://arxiv.org/html/2512.16226v1) — Recent empirical study on compression ratios and Frobenius error trade-offs.
- [Netflix Prize and SVD (PDF)](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) — Classic student paper walking through SVD-based collaborative filtering on the Netflix dataset. **Note:** UPS domain may be unreachable; if so, search "Gower Netflix SVD PDF" for mirrors.
