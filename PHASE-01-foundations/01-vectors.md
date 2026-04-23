# Vectors

**Phase:** PHASE-01-foundations  
**Prerequisites:** None  
**Estimated Time:** 45 minutes

## Why am I learning this?

Because everything in machine learning is a vector. A row in a spreadsheet of customer data? Vector. A word converted to numbers so a model can understand it? Vector. The internal state of a neural network as it processes a sentence? Vector. The weights that define how a model behaves? Vector.

If you do not have an intuitive grasp of what a vector is and what you can do with it, every equation in AI will feel like arbitrary notation. This file exists to make vectors feel obvious.

## Where will I be using it?

- **Embeddings:** Converting words, images, or users into dense vectors is the foundation of modern NLP and recommendation systems. When you hear "embedding space," they mean a space where each point is a vector.
- **Feature vectors:** Every row of training data is a vector of features. `X[i, :]` in NumPy is a feature vector.
- **Weight vectors:** In linear regression and neural networks, the learned parameters form a weight vector $\mathbf{w}$.
- **State vectors:** In reinforcement learning, the current state of the environment is represented as a vector.
- **Attention mechanisms:** In transformers, each token is represented as a vector, and the entire sequence is a matrix of vectors.

## Resources

- [3Blue1Brown: Vectors | Chapter 1, Essence of Linear Algebra](https://www.3blue1brown.com/lessons/vectors) — The single best visual intuition for vectors.
- [Khan Academy: Vectors](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/vectors/v/vector-introduction-linear-algebra) — Algebraic perspective and worked examples.
- [PyTorch Docs: Tensors](https://pytorch.org/docs/stable/tensors.html) — The data structure you will actually use in production.
- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/) — Shows how neural networks "see" vectors internally.

## Appendix

### Notation

- $\mathbf{v} \in \mathbb{R}^n$: A vector with $n$ real-valued components.
- $v_i$: The $i$-th component of $\mathbf{v}$.
- $\mathbf{0}$: The zero vector.

### Row vs. Column Vectors

In strict linear algebra, vectors are column vectors by default ($n 	imes 1$). In NumPy and PyTorch, a 1-D array is shape `(n,)`, which broadcasts like either. Be explicit when shapes matter.

### Common Pitfalls

- Confusing a vector with a scalar.
- Assuming all vectors live in 2D or 3D. In ML, $n$ is usually 128, 768, or 4096.
- Forgetting that the zero vector is a valid vector.

### Further Reading

- [CS231n: Python Numpy Tutorial](https://cs231n.github.io/python-numpy-tutorial/) — How vectors become arrays in practice.
