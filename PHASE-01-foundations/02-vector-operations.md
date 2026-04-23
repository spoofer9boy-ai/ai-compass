# Vector Operations

**Phase:** PHASE-01-foundations  
**Prerequisites:** 01-vectors  
**Estimated Time:** 40 minutes

## Why am I learning this?

Because neural networks do almost nothing except add vectors, scale vectors, and occasionally multiply them in specific ways. The entire training process—gradient descent—is vector addition. A residual connection in a transformer? Vector addition. Averaging word embeddings? Vector addition and scalar multiplication.

If you understand vector addition and scalar multiplication, you understand the mechanical heart of deep learning.

## Where will I be using it?

- **Gradient descent:** Updating weights as $\mathbf{w}_{new} = \mathbf{w}_{old} - \eta 
abla L$ is vector subtraction and scalar multiplication.
- **Residual connections:** $	ext{output} = 	ext{input} + 	ext{transform(input)}$ in ResNets and Transformers.
- **Embedding averaging:** Averaging token embeddings into a sentence embedding.
- **Data augmentation:** Scaling image pixel vectors to change brightness.
- **Attention masking:** Adding a large negative vector to suppress certain positions.

## Resources

- [3Blue1Brown: Linear Combinations, Span, and Basis Vectors](https://www.3blue1brown.com/lessons/span) — Visual intuition for addition and scaling.
- [PyTorch Docs: torch.add](https://pytorch.org/docs/stable/generated/torch.add.html) — Production vector addition.
- [NumPy Docs: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — How vector operations work on arrays of different shapes.

## Appendix

### Notation

- $\mathbf{u} + \mathbf{v}$: Component-wise addition.
- $lpha \mathbf{v}$: Scalar multiplication (stretching or shrinking).
- $\mathbf{u} - \mathbf{v} = \mathbf{u} + (-1)\mathbf{v}$.

### Broadcasting Rules (NumPy/PyTorch)

Two dimensions are compatible when:
1. They are equal, or
2. One of them is 1.

This is why you can add a bias vector to an entire batch of vectors in one line.

### Common Pitfalls

- Adding vectors of different lengths without broadcasting awareness.
- Confusing element-wise multiplication (`*`) with the dot product.
- Forgetting that scalar multiplication changes magnitude but not direction (unless negative).

### Further Reading

- [A Visual Intro to NumPy](http://jalammar.github.io/visual-numpy/) — How vector operations map to array code.
