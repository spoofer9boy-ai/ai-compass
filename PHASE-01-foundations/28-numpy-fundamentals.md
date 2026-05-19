# NumPy Fundamentals

**Phase:** PHASE-01-foundations  
**Prerequisites:** []  
**Estimated Time:** 60 minutes

## Why am I learning this?

You can write machine learning code in pure Python. It will be correct, readable, and unusably slow. Every matrix multiplication, every batch normalization, every convolution—if you implement them with Python loops, you are waiting seconds for what should take milliseconds. NumPy is the bridge between the math you just learned and the code you will actually ship.

NumPy gives you contiguous, typed, multi-dimensional arrays and vectorized operations that run in compiled C and Fortran. It is the memory layout and API that PyTorch, TensorFlow, JAX, and scikit-learn all build on top of. When you call `torch.tensor([...])`, you are using NumPy semantics under the hood. When you load a dataset with `pandas`, the values are stored in NumPy arrays. When you debug a shape mismatch in a transformer, the error message is speaking NumPy dialect.

This file is not a complete API reference. It is the minimal set of ideas—array creation, shape manipulation, broadcasting, indexing, and vectorization—that you need so the rest of the roadmap does not feel like memorizing incantations.

## Where will I be using it?

- **Data Loading:** Every `pd.DataFrame` and `torch.utils.data.Dataset` eventually hands you a NumPy array or a tensor with NumPy-compatible strides.
- **Feature Engineering:** Reshaping image batches, normalizing columns, or computing rolling statistics without writing a single `for` loop.
- **Model Debugging:** Understanding why `matmul` expects `(b, m, k) × (b, k, n)` and what `transpose(0, 2, 1)` actually does to memory layout.
- **Interoperability:** Moving data between PyTorch, TensorFlow, OpenCV, and PIL without copying memory.
- **Custom Kernels:** Writing CUDA extensions or Numba JIT functions that expect C-contiguous NumPy buffers.

## Resources

- [NumPy: The Absolute Beginner's Guide](https://numpy.org/doc/stable/user/absolute_beginners.html) — Official walkthrough of array creation, indexing, and operations.
- [SciPy Lectures: NumPy](https://scipy-lectures.org/intro/numpy/index.html) — Comprehensive, free course on array programming with exercises.
- [Real Python: NumPy Tutorial](https://realpython.com/numpy-tutorial/) — Practical introduction with worked examples and performance comparisons.
- [NumPy Paper (Nature)](https://www.nature.com/articles/s41586-020-2649-2) — The 2020 paper describing NumPy's architecture and role in the scientific Python ecosystem.
- [PyTorch Docs: Tensors](https://pytorch.org/docs/stable/tensors.html) — Official tensor API; note the explicit NumPy compatibility guarantees.

## Appendix

### Notation

- `ndarray`: NumPy's n-dimensional array object.
- `shape`: Tuple describing the size of each dimension, e.g., `(3, 4)`.
- `dtype`: The data type of the array elements, e.g., `float32`, `int64`.
- `axis`: The dimension along which an operation is applied.
- `broadcasting`: The rules that allow NumPy to perform arithmetic on arrays of different shapes.

### Common Pitfalls

- **Copy vs. View:** Slicing with `arr[1:3]` returns a view; advanced indexing `arr[[1, 2]]` returns a copy. Modifying a view modifies the original array.
- **Integer Division:** In Python 3, `/` is float division; in NumPy it is element-wise true division. Use `//` for floor division.
- **Broadcasting Mistakes:** `(3, 1) + (3,)` works; `(3, 2) + (3,)` raises a `ValueError`. Always verify shapes with `.shape` before arithmetic.
- **Memory Layout:** `transpose` and `reshape` often return views, but `flatten` always copies. Use `ravel` when you need a 1-D view and do not intend to modify it.
- **Scalar vs. 0-D Array:** `np.array(5)` is 0-dimensional. You can extract the Python scalar with `.item()`, but mixing 0-D arrays with scalars usually works transparently.

### Further Reading

- [NumPy Broadcasting Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html) — The formal rules with visual diagrams.
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) — Nicolas P. Rougier's free book on vectorization and advanced indexing patterns.
