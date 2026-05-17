# Practice: Attention from Scratch

**Phase:** PHASE-03-deep-learning  
**Subjects Required:** 70 (Self-Attention), 71 (Multi-Head Attention), 72 (Positional Encoding)  
**Estimated Time:** 240 minutes  
**Difficulty:** Advanced

## Industry Context

You are a research engineer at a startup building a lightweight NLP toolkit for edge devices. The team cannot use `torch.nn.MultiheadAttention` because the deployment target is a custom accelerator that only supports basic matrix operations. You need to implement scaled dot-product attention, multi-head attention, and sinusoidal positional encoding from first principles using only NumPy and pure Python, verify numerical equivalence against a reference, and produce a minimal encoder block that can run inference on the device simulator.

## The Problem

Implement a minimal, self-contained attention stack that can process a batch of token sequences and produce context-aware representations. Your solution must include:

1. **Sinusoidal Positional Encoding** — precompute encodings for a max sequence length and embedding dimension, then add them to input embeddings.
2. **Scaled Dot-Product Attention** — compute attention weights from Query, Key, and Value matrices, apply a causal mask for autoregressive decoding, and return both the output and the attention weight matrix.
3. **Multi-Head Attention** — split projected Q/K/V into multiple heads, run scaled dot-product attention in parallel across heads, concatenate, and apply a final linear projection.
4. **A single Transformer Encoder Block** — combine multi-head attention with a feed-forward network, using layer normalization and residual connections.

You are provided with a synthetic dataset and a reference PyTorch implementation. Your NumPy implementation must match the reference outputs within a tolerance of `1e-4` on the provided test cases.

## Constraints

- Do not use `torch.nn.MultiheadAttention`, `torch.nn.Transformer`, or any high-level attention wrapper.
- Do not use `einsum`; use only `np.matmul`, `np.transpose`, `np.reshape`, and basic NumPy operations.
- All weights must be initialized with Xavier uniform (`limit = sqrt(6 / (fan_in + fan_out))`).
- The causal mask must be constructed as a lower-triangular boolean matrix (including the diagonal).
- Must run on a single CPU core and process a batch of 2 sequences of length 8 in under 0.5 seconds.

## Starter Code

```python
import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 2
SEQ_LEN = 8
D_MODEL = 32
NUM_HEADS = 4
D_FF = 64
MAX_SEQ_LEN = 16
SEED = 42

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding:
    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        # TODO: precompute a (max_seq_len, d_model) matrix of sinusoidal encodings.
        # Store it in self.pe.
        self.pe = np.zeros((max_seq_len, d_model))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (batch_size, seq_len, d_model)
        Add positional encodings to x and return the result.
        """
        # TODO
        return x

# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
):
    """
    Q, K, V shapes: (..., seq_len, d_k)
    mask shape:     (seq_len, seq_len) boolean, True where allowed.
                    If None, no masking.

    Returns:
        output: same shape as V
        attn_weights: (..., seq_len, seq_len)
    """
    # TODO
    pass

# ---------------------------------------------------------------------------
# 3. Multi-Head Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: initialize W_q, W_k, W_v, W_o with Xavier uniform.
        # Each is a 2-D numpy array.
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
    ):
        """
        x shape: (batch_size, seq_len, d_model)
        Returns: (batch_size, seq_len, d_model)
        """
        # TODO: project x -> Q, K, V
        # TODO: reshape for multi-head: (batch, heads, seq, d_k)
        # TODO: call scaled_dot_product_attention
        # TODO: concatenate heads and apply W_o
        pass

# ---------------------------------------------------------------------------
# 4. Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        # TODO: Xavier init for W1 (d_model, d_ff) and W2 (d_ff, d_model)
        self.W1 = None
        self.b1 = np.zeros(d_ff)
        self.W2 = None
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: linear -> ReLU -> linear
        pass

# ---------------------------------------------------------------------------
# 5. Layer Normalization
# ---------------------------------------------------------------------------
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: normalize across the last dimension
        pass

# ---------------------------------------------------------------------------
# 6. Transformer Encoder Block
# ---------------------------------------------------------------------------
class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        # TODO: pre-norm or post-norm? Use post-norm for this exercise:
        #       x = ln1(x + mha(x))
        #       x = ln2(x + ff(x))
        pass

# ---------------------------------------------------------------------------
# Reference test harness (do not modify)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = np.random.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).astype(np.float32)

    # Build causal mask
    causal_mask = np.tril(np.ones((SEQ_LEN, SEQ_LEN)), k=0).astype(bool)

    # Your model
    pe = PositionalEncoding(MAX_SEQ_LEN, D_MODEL)
    x_pe = pe.forward(x)

    encoder = TransformerEncoderBlock(D_MODEL, NUM_HEADS, D_FF)
    out = encoder.forward(x_pe, mask=causal_mask)

    print("Output shape:", out.shape)
    print("Output mean:", out.mean())
    print("Output std :", out.std())
```

## Evaluation Criteria

1. **Correctness:** `scaled_dot_product_attention` returns attention weights that sum to 1.0 along the last axis (within `1e-5`) for each query position when a mask is provided.
2. **Shape fidelity:** All intermediate and final tensors have the expected shapes:
   - `pe`: `(MAX_SEQ_LEN, D_MODEL)`
   - `Q/K/V` inside MHA: `(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)`
   - MHA output: `(BATCH_SIZE, SEQ_LEN, D_MODEL)`
   - Encoder block output: `(BATCH_SIZE, SEQ_LEN, D_MODEL)`
3. **Numerical match:** When initialized with the same Xavier weights and fed the same input, your NumPy encoder block output must be within `1e-4` RMSE of the reference PyTorch implementation (provided below in the solution).
4. **Causal masking:** In a forward pass with `mask=None`, the attention weight matrix for a single head must be lower-triangular (allowing positions to attend only to themselves and previous positions) when you construct the causal mask inside `TransformerEncoderBlock.forward` or pass it explicitly.
5. **No banned APIs:** Usage of `torch.nn.MultiheadAttention`, `torch.nn.Transformer`, or `np.einsum` results in automatic failure.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 2
SEQ_LEN = 8
D_MODEL = 32
NUM_HEADS = 4
D_FF = 64
MAX_SEQ_LEN = 16
SEED = 42

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Xavier uniform helper
# ---------------------------------------------------------------------------
def xavier_uniform(fan_in: int, fan_out: int):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

# ---------------------------------------------------------------------------
# 1. Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding:
    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
        position = np.arange(max_seq_len)[:, np.newaxis]          # (max_seq, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )                                                         # (d_model/2,)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)   # (..., seq, seq)

    if mask is not None:
        # mask is True where allowed; we want to set disallowed to -inf
        scores = np.where(mask, scores, -1e9)

    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    output = np.matmul(attn_weights, V)
    return output, attn_weights

# ---------------------------------------------------------------------------
# 3. Multi-Head Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = xavier_uniform(d_model, d_model)
        self.W_k = xavier_uniform(d_model, d_model)
        self.W_v = xavier_uniform(d_model, d_model)
        self.W_o = xavier_uniform(d_model, d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = np.matmul(x, self.W_q)   # (B, seq, d_model)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # Reshape for multi-head: (B, seq, heads, d_k) -> (B, heads, seq, d_k)
        def split_heads(m):
            m = m.reshape(batch_size, seq_len, self.num_heads, self.d_k)
            return np.transpose(m, (0, 2, 1, 3))

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Scaled dot-product attention
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        # attn_out: (B, heads, seq, d_k)

        # Concatenate heads
        attn_out = np.transpose(attn_out, (0, 2, 1, 3))          # (B, seq, heads, d_k)
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)

        # Final linear
        output = np.matmul(attn_out, self.W_o)
        return output

# ---------------------------------------------------------------------------
# 4. Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = xavier_uniform(d_model, d_ff)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = xavier_uniform(d_ff, d_model)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.matmul(x, self.W1) + self.b1) @ self.W2 + self.b2

# ---------------------------------------------------------------------------
# 5. Layer Normalization
# ---------------------------------------------------------------------------
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# ---------------------------------------------------------------------------
# 6. Transformer Encoder Block
# ---------------------------------------------------------------------------
class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        # Self-attention with residual and layer norm
        attn_out = self.mha.forward(x, mask=mask)
        x = self.ln1.forward(x + attn_out)

        # Feed-forward with residual and layer norm
        ff_out = self.ff.forward(x)
        x = self.ln2.forward(x + ff_out)
        return x

# ---------------------------------------------------------------------------
# Reference PyTorch implementation for numerical verification
# ---------------------------------------------------------------------------
def reference_pytorch(x_np, causal_mask_np):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    x = torch.from_numpy(x_np.copy())

    # Positional encoding
    pe = torch.zeros(MAX_SEQ_LEN, D_MODEL)
    position = torch.arange(MAX_SEQ_LEN).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    x = x + pe[:SEQ_LEN, :]

    # A single PyTorch TransformerEncoderLayer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        dim_feedforward=D_FF,
        dropout=0.0,
        batch_first=True,
    )
    # Disable dropout by setting training mode off
    encoder_layer.eval()
    # Convert mask: True -> not masked in PyTorch; we use key_padding_mask style
    # For causal mask we use attn_mask where 0 = attend, -inf = ignore
    attn_mask = torch.from_numpy(~causal_mask_np).float().masked_fill(
        ~torch.from_numpy(causal_mask_np), float('-inf')
    )
    with torch.no_grad():
        out = encoder_layer(x, src_mask=attn_mask)
    return out.numpy()

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = np.random.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).astype(np.float32)
    causal_mask = np.tril(np.ones((SEQ_LEN, SEQ_LEN)), k=0).astype(bool)

    pe = PositionalEncoding(MAX_SEQ_LEN, D_MODEL)
    x_pe = pe.forward(x)

    encoder = TransformerEncoderBlock(D_MODEL, NUM_HEADS, D_FF)
    out = encoder.forward(x_pe, mask=causal_mask)

    print("Output shape:", out.shape)
    print("Output mean:", out.mean())
    print("Output std :", out.std())

    # Numerical check against PyTorch reference
    ref = reference_pytorch(x, causal_mask)
    rmse = np.sqrt(np.mean((out - ref) ** 2))
    print(f"RMSE vs PyTorch reference: {rmse:.2e}")
    assert rmse < 1e-4, f"RMSE too large: {rmse}"
    print("Numerical check PASSED.")
```

</details>

## What You Actually Learned

- **Self-Attention:** You turned a sequence into three projections and used the dot product to let every token decide how much it cares about every other token. The scaling by $\sqrt{d_k}$ prevents softmax from saturating when dimensions grow.
- **Multi-Head Attention:** You split the model dimension into independent subspaces so the network can attend to different syntactic or semantic relationships in parallel. Reshaping with `transpose` and `reshape` is the only mechanical trick; the concept is just running the same attention mechanism $h$ times.
- **Positional Encoding:** You injected absolute position information with sinusoids of varying wavelength, ensuring the model can distinguish "cat" at position 3 from "cat" at position 7 without learning position-specific parameters.
- **Layer Normalization & Residuals:** You stabilized deep signal propagation by normalizing across the feature dimension and adding skip connections. Without these, gradients in a deep stack of attention layers collapse or explode.
- **Causal Masking:** You enforced temporal order in autoregressive settings by zeroing out future positions before softmax. This is the difference between an encoder (bidirectional) and a decoder (left-to-right).

## Sources Used

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Original Transformer paper; defines scaled dot-product attention, multi-head attention, and sinusoidal positional encodings.
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — Line-by-line PyTorch implementation of the paper; used to verify the exact tensor reshapes and masking conventions.
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — Visual walkthrough of how Q, K, V flow through the encoder and decoder stacks.
- [PyTorch Docs: torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — Reference for expected tensor shapes and the default causal masking semantics used in the verification harness.
