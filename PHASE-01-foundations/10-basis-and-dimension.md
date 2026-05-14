# Basis and Dimension

**Phase:** PHASE-01-foundations  
**Prerequisites:** 9  
**Estimated Time:** 40 minutes

## Why am I learning this?

In machine learning you will spend most of your time moving data through high-dimensional vector spaces—embeddings, activations, gradients, weight matrices. A "768-dimensional embedding" sounds concrete, but what matters is not the container size; it is how many *independent directions* your data actually uses. If every sample lies in a 50-dimensional subspace, you are burning compute and memory on 718 redundant coordinates.

Basis and dimension are the tools that let you count those independent directions. A **basis** is a minimal set of vectors that can reconstruct every other vector in the space through linear combination. **Dimension** is simply the size of that minimal set. Once you internalize these ideas, statements like "this matrix has rank 47" or "PCA keeps the top 10 components" stop being magic and become statements about the size of a basis.

You will not write a basis-finding algorithm in production, but you will constantly reason about whether your features are redundant, whether your latent space is collapsed, or whether a layer has enough expressive capacity. That reasoning is impossible without knowing what a basis is and why all bases for the same space have the same length.

## Where will I be using it?

- **Embedding spaces:** Word and document embeddings live in vector spaces. The intrinsic dimension of your data determines how many parameters you actually need to represent it meaningfully.
- **Dimensionality reduction (PCA):** PCA finds a new basis (eigenvectors) and discards dimensions with low variance. You need to understand what a basis is to see why this transformation preserves the structure of the data.
- **Computer graphics:** Objects are transformed between model space, world space, and camera space. Each space is just a different basis for $\mathbb{R}^3$; the change-of-basis matrix moves coordinates from one to another.
- **Model compression:** Low-rank factorization exploits the fact that a weight matrix may have an effective basis much smaller than its literal dimensions, letting you store two smaller matrices instead of one large one.
- **Feature engineering:** If one feature is an exact linear combination of others, the feature set is linearly dependent. Removing redundancy is the same thing as shrinking to a smaller basis.

## Resources

- [MIT OCW: Basis and Dimension](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/basis-and-dimension/) — Lecture notes and video from Gilbert Strang's 18.06 course.
- [3Blue1Brown: Linear combinations, span, and basis vectors](https://www.3blue1brown.com/lessons/span/) — Visual intuition for why a basis acts as a "coordinate system" for a space.
- [Interactive Linear Algebra: Basis and Dimension](https://textbooks.math.gatech.edu/ila/dimension.html) — Free textbook chapter with proofs and worked examples.
- [Khan Academy: Basis of a subspace](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/subspace-basis/v/linear-algebra-basis-of-a-subspace) — Step-by-step video building the definition from span and independence.
- [Wikipedia: Basis (linear algebra)](<https://en.wikipedia.org/wiki/Basis_(linear_algebra)>) — Rigorous definition and key theorems, including the fact that all bases share the same cardinality.

## Appendix

### Notation

- $V$: a vector space.
- $\mathcal{B} = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$: a basis for $V$.
- $\dim(V) = n$: the dimension of $V$, equal to the number of vectors in any basis.

### Key Theorems

- **Basis Theorem:** Every vector space has a basis (in the finite-dimensional case this follows directly from the spanning-set reduction process).
- **Dimension Theorem:** All bases for a given vector space contain the same number of vectors. Therefore dimension is well-defined and does not depend on which basis you choose.
- **Span + Independence = Basis:** A set of vectors is a basis if and only if it is linearly independent and spans the entire space.

### Common Pitfalls

- **Confusing a basis with a spanning set:** A spanning set can be larger than a basis; a basis is *minimal*. If you have five vectors spanning $\mathbb{R}^3$, only three of them form a basis.
- **Assuming the standard basis is the only basis:** $\mathbb{R}^n$ has infinitely many bases. The standard basis $\{\mathbf{e}_1, \dots, \mathbf{e}_n\}$ is simply the most convenient coordinate system, not the only one.
- **Ignoring the zero vector space:** The space $\{\mathbf{0}\}$ has dimension $0$ and its basis is the empty set. This edge case shows up when talking about kernels (nullspaces) of full-rank matrices.

### Further Reading

- [MIT OCW Lecture 9 PDF Summary](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/0bbc30e3f1d7933ea07a2d2e9ab050d9_MIT18_06SCF11_Ses1.9sum.pdf) — One-page summary of independence, basis, and dimension.
