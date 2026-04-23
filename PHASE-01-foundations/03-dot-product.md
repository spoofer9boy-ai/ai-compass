# Dot Product

**Phase:** PHASE-01-foundations  
**Prerequisites:** 01-vectors, 02-vector-operations  
**Estimated Time:** 50 minutes

## Why am I learning this?

Because the dot product is the Swiss Army knife of machine learning. It measures alignment between two vectors. High dot product means "these point in similar directions." Low (or negative) means "these are unrelated or opposite."

When you search for a document semantically, when a transformer decides which tokens to attend to, when a recommender predicts if a user will like an item—all of these are dot products in disguise.

## Where will I be using it?

- **Cosine similarity:** The core operation in semantic search and RAG systems. $	ext{sim}(\mathbf{a}, \mathbf{b}) = rac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \, ||\mathbf{b}||}$.
- **Attention mechanisms:** In transformers, attention scores are computed as dot products between query and key vectors.
- **Linear regression (closed form):** The normal equation $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is built from dot products.
- **Neural network layers:** A fully connected layer is a matrix of dot products between input vector and each neuron's weight vector.
- **Recommendation systems:** Matrix factorization models factor the user-item rating matrix into dot products of latent vectors.

## Resources

- [3Blue1Brown: Dot Products and Duality](https://www.3blue1brown.com/lessons/dot-products) — Geometric intuition that changes how you see ML.
- [Stanford CS231n: Linear Algebra Review](https://cs231n.stanford.edu/linear-algebra-review.pdf) — Concise reference with ML focus.
- [PyTorch Docs: torch.dot](https://pytorch.org/docs/stable/generated/torch.dot.html) — The API.
- [Pinecone: Similarity Measures](https://www.pinecone.io/learn/vector-similarity/) — How dot product becomes search.

## Appendix

### Notation

- $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$
- $\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}|| \, ||\mathbf{b}|| \cos(	heta)$

### Geometric Interpretation

The dot product projects one vector onto another and scales by the magnitude. This is why it measures alignment.

### From Dot Product to Cosine Similarity

Cosine similarity normalizes the dot product by magnitudes, removing the effect of vector length. This matters in search: a long document and a short query should be comparable.

### Common Pitfalls

- Using raw dot product when cosine similarity is more appropriate.
- Forgetting that dot product is only defined for vectors of the same dimension.
- Confusing dot product with element-wise product.

### Further Reading

- [Blog: The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — See dot products powering attention.
