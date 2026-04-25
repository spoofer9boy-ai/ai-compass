# Practice: Image Compression with SVD

**Phase:** PHASE-01-foundations  
**Subjects Required:** 11-eigenvalues, 12-eigenvectors, 13-singular-value-decomposition  
**Estimated Time:** 120 minutes  
**Difficulty:** Beginner

## Industry Context

You are the imaging engineer at a remote-sensing startup. Your satellites capture 4096×4096 grayscale terrain photos, but the downlink radio has limited bandwidth. The team needs a lossy compression method that:

1. Runs deterministically on the satellite's CPU (no training data, no neural network).
2. Preserves large terrain features (coastlines, mountain ridges) even at aggressive compression.
3. Allows mission control to request a specific reconstruction quality by tuning a single integer parameter.

Singular Value Decomposition (SVD) is the classical tool for this. By keeping only the top-k singular values, you discard fine-grained noise and texture while retaining the dominant structural information. This technique predates deep learning and is still used in scientific imaging where interpretability and stability matter more than neural compression ratios.

## The Problem

Implement a `SVDCompressor` class that compresses a grayscale image represented as a 2D NumPy array using low-rank SVD approximation.

Your implementation must:

1. Accept an image matrix `A` of shape `(m, n)`.
2. Compute the SVD: `A = U Σ V^T`.
3. Reconstruct a rank-k approximation `A_k = U_k Σ_k V_k^T` using only the top `k` singular values.
4. Report:
   - **Relative reconstruction error** in Frobenius norm: `||A - A_k||_F / ||A||_F`
   - **Compression ratio**: `original_bytes / compressed_bytes`

The compressed representation should store only:
- The first `k` columns of `U` (shape `m × k`)
- The first `k` singular values (shape `k`)
- The first `k` rows of `V^T` (shape `k × n`)

## Constraints

- Use **only NumPy**. No PIL, OpenCV, scikit-image, or pre-built compression libraries for the core logic.
- You may assume the input is already a 2D NumPy array of dtype `float64`.
- `k` must be validated: `1 ≤ k ≤ min(m, n)`.
- The reconstruction must use the standard SVD low-rank formula `U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]`.
- The Frobenius norm must be computed with `np.linalg.norm`, not manually summed.

## Starter Code

```python
import numpy as np
from typing import Tuple

class SVDCompressor:
    def __init__(self, image: np.ndarray):
        """
        image: 2D NumPy array of shape (m, n) representing a grayscale image.
        """
        if image.ndim != 2:
            raise ValueError("Image must be a 2D array.")
        self.image = image.astype(np.float64)
        self.U = None
        self.s = None
        self.Vt = None

    def decompose(self):
        """
        TODO: Compute the full SVD of self.image.
        Store the results in self.U, self.s, self.Vt.
        Use numpy.linalg.svd with full_matrices=False.
        """
        pass

    def reconstruct(self, k: int) -> np.ndarray:
        """
        TODO: Return the rank-k approximation of the image.
        Validate k is in the valid range.
        """
        pass

    def relative_error(self, k: int) -> float:
        """
        TODO: Compute ||A - A_k||_F / ||A||_F.
        """
        pass

    def compression_ratio(self, k: int) -> float:
        """
        TODO: Compute the ratio of original storage to compressed storage.

        Original storage: m * n floats.
        Compressed storage: (m * k) + k + (k * n) floats.
        """
        pass

    def compress_and_report(self, k: int) -> dict:
        """
        Run reconstruction, error, and ratio. Return a dict with keys:
        'reconstructed', 'relative_error', 'compression_ratio'.
        """
        return {
            'reconstructed': self.reconstruct(k),
            'relative_error': self.relative_error(k),
            'compression_ratio': self.compression_ratio(k),
        }


# --- Test harness ---
if __name__ == "__main__":
    # Create a synthetic test image with clear structure
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)
    image = np.sin(10 * X) * np.cos(10 * Y) + 0.5 * np.random.randn(256, 256)

    compressor = SVDCompressor(image)
    compressor.decompose()

    for k in [5, 20, 50]:
        result = compressor.compress_and_report(k)
        print(f"k={k:3d} | error={result['relative_error']:.4f} | ratio={result['compression_ratio']:.2f}x")
```

## Evaluation Criteria

1. **Correctness:** `reconstruct(k)` must exactly equal `U[:, :k] @ diag(s[:k]) @ Vt[:k, :]`.
2. **Error metric:** `relative_error(k)` should decrease monotonically as `k` increases (for the test image above, `k=5` should yield error > 0.3, `k=50` should yield error < 0.15).
3. **Compression ratio:** For a 256×256 image with `k=20`, the ratio should be approximately `256*256 / (256*20 + 20 + 20*256) ≈ 6.3x`.
4. **Edge handling:** Calling `reconstruct(0)` or `reconstruct(min(m,n)+1)` should raise a clear `ValueError`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
from typing import Tuple

class SVDCompressor:
    def __init__(self, image: np.ndarray):
        if image.ndim != 2:
            raise ValueError("Image must be a 2D array.")
        self.image = image.astype(np.float64)
        self.m, self.n = self.image.shape
        self.U = None
        self.s = None
        self.Vt = None

    def decompose(self):
        """
        Compute the full SVD: A = U @ diag(s) @ Vt
        full_matrices=False gives the "economy" SVD:
        U is (m, r), s is (r,), Vt is (r, n) where r = min(m, n).
        """
        self.U, self.s, self.Vt = np.linalg.svd(self.image, full_matrices=False)

    def reconstruct(self, k: int) -> np.ndarray:
        """
        Reconstruct the rank-k approximation.
        A_k = U_k @ Σ_k @ V_k^T
        """
        if not (1 <= k <= min(self.m, self.n)):
            raise ValueError(f"k must be between 1 and {min(self.m, self.n)}, got {k}")

        # Extract the top-k components
        Uk = self.U[:, :k]
        sk = self.s[:k]
        Vtk = self.Vt[:k, :]

        # Reconstruct: U_k @ diag(s_k) @ V_k^T
        return Uk @ np.diag(sk) @ Vtk

    def relative_error(self, k: int) -> float:
        """
        Relative Frobenius norm error: ||A - A_k||_F / ||A||_F
        """
        Ak = self.reconstruct(k)
        error_norm = np.linalg.norm(self.image - Ak, 'fro')
        original_norm = np.linalg.norm(self.image, 'fro')
        if original_norm == 0:
            return 0.0
        return error_norm / original_norm

    def compression_ratio(self, k: int) -> float:
        """
        Ratio of original float count to compressed float count.
        Original: m * n
        Compressed: m*k (U_k) + k (singular values) + k*n (V_k^T)
        """
        original = self.m * self.n
        compressed = (self.m * k) + k + (k * self.n)
        return original / compressed

    def compress_and_report(self, k: int) -> dict:
        return {
            'reconstructed': self.reconstruct(k),
            'relative_error': self.relative_error(k),
            'compression_ratio': self.compression_ratio(k),
        }


# --- Test harness ---
if __name__ == "__main__":
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)
    image = np.sin(10 * X) * np.cos(10 * Y) + 0.5 * np.random.randn(256, 256)

    compressor = SVDCompressor(image)
    compressor.decompose()

    for k in [5, 20, 50]:
        result = compressor.compress_and_report(k)
        print(f"k={k:3d} | error={result['relative_error']:.4f} | ratio={result['compression_ratio']:.2f}x")
```

</details>

## What You Actually Learned

- **Eigenvalues:** The singular values `σ_i` are the square roots of the eigenvalues of `A^T A`. They rank the amount of "energy" or information each principal axis carries. By truncating small singular values, you discard the axes that contribute least to the image structure.
- **Eigenvectors:** The columns of `U` are the eigenvectors of `A A^T` (left singular vectors); the rows of `V^T` are the eigenvectors of `A^T A` (right singular vectors). They form the orthonormal bases in which the image is diagonal.
- **Singular Value Decomposition:** You decomposed a matrix into three interpretable factors and rebuilt it with fewer parameters. This is the exact same mathematics that powers PCA, recommendation systems, and latent semantic analysis—only the data changes.
- **Industry Reality:** Low-rank approximation is not just a classroom trick. It is used in:
  - **Hyperspectral imaging** to reduce sensor noise.
  - **Video compression** for background subtraction (the background is low-rank, the foreground is sparse).
  - **Recommender systems** where the user-item rating matrix is approximated by a low-rank factorization.
