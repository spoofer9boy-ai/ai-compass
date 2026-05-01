# Vector Spaces

**Phase:** PHASE-01-foundations  
**Prerequisites:** 01-vectors, 02-vector-operations  
**Estimated Time:** 45 minutes

## Why am I learning this?

Because "vector" is not just a data type—it is a member of a space with rules. When you train a model, you are not just shuffling numbers around; you are moving points inside a vector space. Understanding what makes a space a *vector space* tells you which operations are legal, which transformations preserve structure, and why certain algorithms fail when you break the rules.

You will never write `is_vector_space(V)` in production code. But you will spend hours debugging why PCA gives nonsense (your data does not live in a well-behaved subspace), why your embeddings drift during training (the space is not being preserved), or why a dimensionality reduction technique destroys information you care about (it projected onto the wrong subspace). Vector spaces are the lens that makes those bugs obvious.

A vector space is any set of objects where addition and scalar multiplication behave the way you expect: addition is commutative and associative, there is a zero vector, every vector has an additive inverse, and scaling distributes over addition. These rules seem trivial until you realize that embeddings, gradients, features, and activations all rely on them. Break the rules—say, by adding two one-hot encodings as if they were vectors in a continuous space—and your model will learn garbage without throwing an error.

## Where will I be using it?

- **Embedding spaces:** Word2Vec, BERT, and CLIP all map data into vector spaces where semantic relationships become geometric relationships. "King − Man + Woman ≈ Queen" works because those words live in the same vector space and the arithmetic respects the space's structure.
- **Feature engineering:** Every feature vector is a point in $\mathbb{R}^n$. Understanding that space tells you why scaling features matters (it changes the geometry) and why missing values break assumptions (they remove a point from the space).
- **Subspaces in deep learning:** The output of a linear layer is a subspace of the input space. Dropout randomly zeros neurons, which is equivalent to projecting onto random subspaces during training. BatchNorm re-centers and rescales the data within its feature subspace.
- **Dimensionality reduction:** PCA, t-SNE, and UMAP all assume your data lies on or near a lower-dimensional subspace of a larger vector space. If that assumption is wrong, the visualization is misleading.
- **Control theory and robotics:** State spaces in reinforcement learning and dynamical systems are vector spaces. Transitions are linear operators acting on those spaces.

## Resources

- [3Blue1Brown: Linear Combinations, Span, and Basis Vectors](https://www.3blue1brown.com/lessons/span) — Visual intuition for why closure under addition and scaling defines a space.
- [Khan Academy: Vectors and Spaces](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces) — Interactive exercises on subspaces, spans, and the formal axioms.
- [MIT OpenCourseWare: 18.06 Linear Algebra (Spring 2010)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Gilbert Strang's lectures on vector spaces and the four fundamental subspaces.
- [PyTorch Docs: Tensors](https://pytorch.org/docs/stable/tensors.html) — The data structure that implements vector spaces in practice; note how operations preserve shape and broadcasting rules.

## Appendix

### Notation

- $\mathbb{R}^n$: The standard $n$-dimensional vector space of real-valued vectors.
- $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$: Vectors belonging to a vector space $V$.
- $c \in \mathbb{R}$: A scalar used for scalar multiplication.
- $\mathbf{0}$: The zero vector, the additive identity of the space.
- $\text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$: The set of all linear combinations of the vectors $\mathbf{v}_1, \dots, \mathbf{v}_k$; the smallest subspace containing them.

### The Axioms (What Actually Defines a Vector Space)

For a set $V$ with addition and scalar multiplication to be a vector space, the following must hold for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $a, b$:

1. **Closure under addition:** $\mathbf{u} + \mathbf{v} \in V$.
2. **Commutativity:** $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.
3. **Associativity:** $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$.
4. **Additive identity:** There exists $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
5. **Additive inverses:** For every $\mathbf{v}$, there exists $-\mathbf{v}$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.
6. **Closure under scalar multiplication:** $a\mathbf{v} \in V$.
7. **Distributivity (vector):** $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$.
8. **Distributivity (scalar):** $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$.
9. **Associativity of scalars:** $a(b\mathbf{v}) = (ab)\mathbf{v}$.
10. **Multiplicative identity:** $1\mathbf{v} = \mathbf{v}$.

In machine learning, you almost always work in $\mathbb{R}^n$ or subspaces of it, so these axioms are satisfied automatically. But when you move to function spaces (kernel methods, Gaussian processes) or abstract spaces (group representations), these axioms become the difference between a valid operation and undefined nonsense.

### Subspaces

A **subspace** $W$ of a vector space $V$ is a subset that is itself a vector space under the same operations. To check if $W$ is a subspace, you only need to verify three things:

1. $\mathbf{0} \in W$.
2. $W$ is closed under addition.
3. $W$ is closed under scalar multiplication.

In deep learning, the column space of a weight matrix is a subspace. If your weight matrix has rank $r < n$, its outputs are constrained to an $r$-dimensional subspace of $\mathbb{R}^n$. That is why rank deficiency causes information loss.

### Common Pitfalls

- **Treating every collection of vectors as a vector space.** A set of one-hot vectors is not closed under addition, so it is not a subspace. If you average one-hot vectors, you leave the set of valid one-hots and enter the continuous simplex.
- **Assuming all spaces are Euclidean.** The dot product gives angles and distances, but a vector space alone does not define them. Those require an *inner product space*, a stricter structure.
- **Confusing dimension with cardinality.** The dimension of a space is the size of a basis, not the number of vectors in it. $\mathbb{R}^3$ contains infinitely many vectors but has dimension 3.
- **Forgetting that the zero vector must be included.** Any valid subspace must contain $\mathbf{0}$. A plane not passing through the origin is not a subspace (it is an affine space).

### Further Reading

- [Wikipedia: Vector Space](https://en.wikipedia.org/wiki/Vector_space) — Formal definition with examples across mathematics and physics.
- [CS231n: Python Numpy Tutorial](https://cs231n.github.io/python-numpy-tutorial/) — Practical introduction to how vector spaces are represented in array libraries.
