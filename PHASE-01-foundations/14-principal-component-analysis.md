# Principal Component Analysis

**Phase:** PHASE-01-foundations  
**Prerequisites:** 13 (Singular Value Decomposition)  
**Estimated Time:** 50 minutes

## Why am I learning this?

You will rarely train a model on raw 10,000-dimensional feature vectors. Production datasets have redundant sensors, correlated measurements, and noisy columns that bloat memory and slow training. Principal Component Analysis (PCA) is the first tool engineers reach for to cut through that bloat without throwing away the signal.

PCA is not magic. It is a disciplined projection: find the directions where your data actually varies, keep the few that matter, and drop the rest. The result is a lower-dimensional representation that preserves as much variance as possible. In practice, this means you can compress embeddings, visualize high-dimensional clusters, and speed up downstream models with a single linear transform.

You already have the machinery for it. PCA is Singular Value Decomposition (SVD) applied to a centered data matrix. If you understand eigenvectors, eigenvalues, and SVD, PCA is a one-line conceptual step: diagonalize the covariance matrix, sort by eigenvalue magnitude, and truncate. The hard part is knowing when it helps, when it hurts, and how to interpret the components you keep.

## Where will I be using it?

- **Embedding compression:** Reduce 768-dimensional sentence embeddings to 128 for faster nearest-neighbor search in retrieval systems.
- **Visualization:** Project high-dimensional image or text features into 2D/3D for clustering inspection and debugging.
- **Denoising:** Reconstruct clean data from a truncated SVD of a noisy matrix (e.g., collaborative filtering before matrix factorization).
- **Preprocessing pipelines:** Feed PCA-reduced features into classical models (logistic regression, SVM, random forests) to reduce collinearity and training time.
- **Face recognition:** Eigenfaces, an early computer-vision technique, are literally the top principal components of a face-image dataset.

## Resources

- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — The API you will actually use. Covers `fit`, `transform`, `inverse_transform`, and `explained_variance_ratio_`.
- [CS 357 Textbook: PCA](https://cs357.cs.illinois.edu/textbook/notes/pca.html) — Clean, rigorous derivation from covariance matrices to SVD, with worked examples.
- [IBM: What Is Principal Component Analysis?](https://www.ibm.com/think/topics/principal-component-analysis) — Practical overview of when and why to apply PCA in business contexts.
- [Wikipedia: Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) — Comprehensive reference on mathematical properties, variants, and history.
- [arXiv: SO(3)-invariant PCA](https://arxiv.org/abs/2510.18827) — Example of modern PCA extensions in molecular and geometric data; shows how the core idea propagates into research.

## Appendix

### Notation

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: Data matrix with $n$ samples and $d$ features, mean-centered.
- $\mathbf{C} = \frac{1}{n-1} \mathbf{X}^\top \mathbf{X}$: Sample covariance matrix.
- $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\top$: Eigendecomposition of $\mathbf{C}$.
- $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$: SVD of the data matrix.
- $\mathbf{Z} = \mathbf{X} \mathbf{V}_k$: Projected data onto top-$k$ principal components.

### Common Pitfalls

- **Forgetting to center the data:** PCA is not rotationally invariant around the origin. Always subtract the mean before computing the covariance matrix or SVD.
- **Confusing PCA with feature selection:** PCA creates new synthetic features (linear combinations of old ones). You lose interpretability of individual original columns.
- **Blindly keeping $k$ components:** Use `explained_variance_ratio_` to decide $k$. A common rule is to keep enough components to capture 95% of total variance, but domain knowledge should override this.
- **Applying PCA to categorical data:** PCA assumes continuous, real-valued inputs. One-hot encoded categoricals will not behave well without preprocessing.

### Further Reading

- [Distill.pub: PCA](https://distill.pub) — Interactive visualizations of dimensionality reduction (check for PCA-specific articles).
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues) — Visual intuition for the core machinery behind PCA.
