# Practice: RNN Text Generator

**Phase:** PHASE-03-deep-learning  
**Subjects Required:** 67 (Recurrent Neural Networks), 68 (LSTM and GRU)  
**Estimated Time:** 210 minutes  
**Difficulty:** Intermediate

## Industry Context

You are an ML engineer at a content-creation startup. The product team wants a lightweight, on-device text-completion assistant that can suggest the next few words as a user types on a mobile keyboard. Running a full Transformer in the browser or on a phone is too heavy for latency and battery reasons. A small recurrent model—trained on public domain books and fine-tuned on the user's own notes—can run locally with acceptable quality. Your task is to prototype the core text-generation engine using only vanilla NumPy and the concepts from the RNN and LSTM subjects.

## The Problem

Implement a character-level text generator using a stacked LSTM network. You will:

1. Build a vocabulary from a text corpus.
2. Implement forward propagation for a multi-layer LSTM from scratch.
3. Train the network with truncated backpropagation through time (BPTT) to predict the next character.
4. Sample new text from the trained model using a temperature-controlled softmax.

You must implement the LSTM cell equations, the stacked forward pass, and the sampling loop yourself. You may use NumPy for array operations, but you may not use PyTorch, TensorFlow, JAX, or any auto-diff framework.

## Constraints

- Do not use PyTorch, TensorFlow, JAX, Keras, or any deep-learning framework. Use only NumPy.
- The model must support at least 2 LSTM layers and a hidden size of at least 128.
- Training must use truncated BPTT with a sequence length of 100 characters.
- Sampling must support a `temperature` parameter that scales logits before the softmax.
- The script must run on a single CPU core and complete a small demo training loop (1,000 iterations) in under 5 minutes on a modern laptop.

## Starter Code

```python
import numpy as np

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_corpus():
    """Return a small public-domain text for demo training."""
    return (
        "It was the best of times, it was the worst of times, "
        "it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity, "
        "it was the season of Light, it was the season of Darkness, "
        "it was the spring of hope, it was the winter of despair, "
        "we had everything before us, we had nothing before us, "
        "we were all going direct to Heaven, we were all going direct "
        "the other way—in short, the period was so far like the present "
        "period, that some of its noisiest authorities insisted on its "
        "being received, for good or for evil, in the superlative degree "
        "of comparison only."
    )

def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

def encode(text, char_to_idx):
    return np.array([char_to_idx[ch] for ch in text], dtype=np.int32)

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # x shape: (vocab_size,)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# ---------------------------------------------------------------------------
# LSTM cell (single time step)
# ---------------------------------------------------------------------------

def lstm_step(x_t, h_prev, c_prev, Wxh, Whh, b):
    """
    Single LSTM forward step.

    Parameters
    ----------
    x_t : ndarray of shape (input_size,)
    h_prev : ndarray of shape (hidden_size,)
    c_prev : ndarray of shape (hidden_size,)
    Wxh : ndarray of shape (input_size, 4*hidden_size)
    Whh : ndarray of shape (hidden_size, 4*hidden_size)
    b : ndarray of shape (4*hidden_size,)

    Returns
    -------
    h_next, c_next, cache
    """
    # TODO: implement the LSTM gate equations.
    # Concatenate x_t and h_prev for a single matrix multiplication if you prefer.
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Stacked LSTM forward pass
# ---------------------------------------------------------------------------

def forward_sequence(inputs, params):
    """
    Run a sequence of one-hot vectors through a stacked LSTM.

    Parameters
    ----------
    inputs : list of ndarray, length seq_len, each shape (vocab_size,)
    params : dict with keys 'layers' (list of dicts) and 'Wy', 'by'

    Returns
    -------
    logits : ndarray of shape (seq_len, vocab_size)
    caches : list of per-layer, per-time-step caches for backprop
    """
    # TODO: iterate over time steps, then over layers.
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Loss and BPTT
# ---------------------------------------------------------------------------

def cross_entropy(logits, targets):
    """
    logits : ndarray of shape (seq_len, vocab_size)
    targets : ndarray of shape (seq_len,), integer indices
    """
    # TODO: compute average cross-entropy over the sequence.
    raise NotImplementedError

def bptt(inputs, targets, params, caches, logits, clip_value=5.0):
    """
    Truncated backpropagation through time.

    Returns gradients as a dict mirroring `params`.
    """
    # TODO: backpropagate through time and layers.
    # Clip gradients element-wise to `clip_value`.
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample(params, seed_string, char_to_idx, idx_to_char, n=200, temperature=1.0):
    """
    Generate `n` new characters conditioned on `seed_string`.
    """
    # TODO: run the seed through the network to warm up hidden states,
    # then sample one character at a time using temperature scaling.
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    text = load_corpus()
    chars, char_to_idx, idx_to_char = build_vocab(text)
    data = encode(text, char_to_idx)
    vocab_size = len(chars)

    hyper = {
        'hidden_size': 128,
        'num_layers': 2,
        'seq_len': 100,
        'learning_rate': 0.01,
        'num_iterations': 1000,
    }

    # TODO: initialise parameters (Xavier/He style small random values).
    # TODO: run the training loop:
    #       1. Sample a random chunk of length `seq_len` from `data`.
    #       2. One-hot encode inputs and targets.
    #       3. Forward pass -> logits.
    #       4. Compute loss.
    #       5. BPTT -> gradients.
    #       6. SGD update.
    #       7. Every 100 iterations, print loss and sample text.

if __name__ == "__main__":
    train()
```

## Evaluation Criteria

1. **Correctness:** The LSTM gates (forget, input, candidate, output) match the standard equations. The forward pass produces logits of shape `(seq_len, vocab_size)`. The loss decreases over training.
2. **Efficiency:** The demo loop (1,000 iterations, 2-layer LSTM, hidden size 128, sequence length 100) finishes in under 5 minutes on a single CPU core.
3. **Sampling quality:** After training, `sample(..., temperature=0.5)` produces text that respects local spelling and punctuation patterns from the corpus (e.g., spaces after commas, capitalisation of "It").
4. **Gradient clipping:** Gradients are clipped element-wise to prevent explosion.
5. **No frameworks:** No import of torch, tensorflow, jax, or keras. Only NumPy and the Python standard library.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_corpus():
    return (
        "It was the best of times, it was the worst of times, "
        "it was the age of wisdom, it was the age of foolishness, "
        "it was the epoch of belief, it was the epoch of incredulity, "
        "it was the season of Light, it was the season of Darkness, "
        "it was the spring of hope, it was the winter of despair, "
        "we had everything before us, we had nothing before us, "
        "we were all going direct to Heaven, we were all going direct "
        "the other way—in short, the period was so far like the present "
        "period, that some of its noisiest authorities insisted on its "
        "being received, for good or for evil, in the superlative degree "
        "of comparison only."
    )

def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

def encode(text, char_to_idx):
    return np.array([char_to_idx[ch] for ch in text], dtype=np.int32)

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# ---------------------------------------------------------------------------
# LSTM cell (single time step)
# ---------------------------------------------------------------------------

def lstm_step(x_t, h_prev, c_prev, Wxh, Whh, b):
    """
    Standard LSTM forward step.
    """
    z = np.dot(x_t, Wxh) + np.dot(h_prev, Whh) + b
    hidden_size = h_prev.shape[0]
    z_i = z[0:hidden_size]
    z_f = z[hidden_size:2*hidden_size]
    z_g = z[2*hidden_size:3*hidden_size]
    z_o = z[3*hidden_size:4*hidden_size]

    i = sigmoid(z_i)
    f = sigmoid(z_f)
    g = np.tanh(z_g)
    o = sigmoid(z_o)

    c_next = f * c_prev + i * g
    h_next = o * np.tanh(c_next)

    cache = (x_t, h_prev, c_prev, i, f, g, o, c_next, Wxh, Whh, b)
    return h_next, c_next, cache

# ---------------------------------------------------------------------------
# Stacked LSTM forward pass
# ---------------------------------------------------------------------------

def forward_sequence(inputs, params):
    """
    inputs : list of ndarray, length seq_len, each shape (vocab_size,)
    """
    layers = params['layers']
    Wy = params['Wy']
    by = params['by']
    num_layers = len(layers)
    seq_len = len(inputs)

    # Initialise hidden and cell states to zero
    h_states = [[np.zeros_like(l['b'][:l['b'].shape[0]//4]) for _ in range(seq_len)] for l in layers]
    c_states = [[np.zeros_like(l['b'][:l['b'].shape[0]//4]) for _ in range(seq_len)] for l in layers]
    caches = [[] for _ in layers]

    for t in range(seq_len):
        x_t = inputs[t]
        for l_idx, layer in enumerate(layers):
            h_prev = h_states[l_idx][t-1] if t > 0 else np.zeros_like(layer['b'][:layer['b'].shape[0]//4])
            c_prev = c_states[l_idx][t-1] if t > 0 else np.zeros_like(layer['b'][:layer['b'].shape[0]//4])
            h_next, c_next, cache = lstm_step(x_t, h_prev, c_prev, layer['Wxh'], layer['Whh'], layer['b'])
            h_states[l_idx][t] = h_next
            c_states[l_idx][t] = c_next
            caches[l_idx].append(cache)
            x_t = h_next  # output of this layer is input to next

    # Output layer: use top-layer hidden states
    logits = np.stack([np.dot(h_states[-1][t], Wy) + by for t in range(seq_len)], axis=0)
    return logits, caches

# ---------------------------------------------------------------------------
# Loss and BPTT
# ---------------------------------------------------------------------------

def cross_entropy(logits, targets):
    seq_len, vocab_size = logits.shape
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    log_probs = -np.log(probs[np.arange(seq_len), targets] + 1e-12)
    return np.mean(log_probs)

def bptt(inputs, targets, params, caches, logits, clip_value=5.0):
    seq_len, vocab_size = logits.shape
    layers = params['layers']
    num_layers = len(layers)
    hidden_size = layers[0]['b'].shape[0] // 4

    # Softmax gradient
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    dy = probs.copy()
    dy[np.arange(seq_len), targets] -= 1
    dy /= seq_len

    # Gradients for output layer
    h_top = np.stack([caches[-1][t][1] for t in range(seq_len)], axis=0)  # h_prev from top layer caches is not stored; recompute is expensive. Instead we re-run a lightweight cache.
    # To keep code simple and self-contained, we re-use the forward hidden states by recomputing them in a real implementation.
    # Here we approximate by extracting h from cache[1] (h_prev) — but for t we need h_next.
    # Simplification: we will re-run forward in a full solution or store h_states. For this educational solution we store h_states globally or recompute.
    # Below is a corrected version that stores h_states during forward and passes them in.
    raise NotImplementedError("See full solution in repository for complete BPTT with stored hidden states.")

# Because the starter code intentionally leaves BPTT as an exercise, the full
# solution below includes a complete, runnable implementation with stored states.

# ---------------------------------------------------------------------------
# Full runnable solution (replaces the stub above)
# ---------------------------------------------------------------------------

def init_params(vocab_size, hidden_size, num_layers):
    params = {'layers': []}
    for l in range(num_layers):
        input_dim = vocab_size if l == 0 else hidden_size
        Wxh = np.random.randn(input_dim, 4 * hidden_size) * 0.01
        Whh = np.random.randn(hidden_size, 4 * hidden_size) * 0.01
        b = np.zeros(4 * hidden_size)
        # forget gate bias to 1 for better gradient flow
        b[hidden_size:2*hidden_size] = 1.0
        params['layers'].append({'Wxh': Wxh, 'Whh': Whh, 'b': b})
    params['Wy'] = np.random.randn(hidden_size, vocab_size) * 0.01
    params['by'] = np.zeros(vocab_size)
    return params

def forward_sequence_full(inputs, params):
    layers = params['layers']
    num_layers = len(layers)
    seq_len = len(inputs)
    hidden_size = layers[0]['b'].shape[0] // 4

    h_states = []
    c_states = []
    caches = [[] for _ in layers]

    for l in range(num_layers):
        h_states.append([np.zeros(hidden_size) for _ in range(seq_len)])
        c_states.append([np.zeros(hidden_size) for _ in range(seq_len)])

    for t in range(seq_len):
        x_t = inputs[t]
        for l_idx in range(num_layers):
            h_prev = h_states[l_idx][t-1] if t > 0 else np.zeros(hidden_size)
            c_prev = c_states[l_idx][t-1] if t > 0 else np.zeros(hidden_size)
            layer = layers[l_idx]
            h_next, c_next, cache = lstm_step(x_t, h_prev, c_prev, layer['Wxh'], layer['Whh'], layer['b'])
            h_states[l_idx][t] = h_next
            c_states[l_idx][t] = c_next
            caches[l_idx].append(cache)
            x_t = h_next

    logits = np.stack([np.dot(h_states[-1][t], params['Wy']) + params['by'] for t in range(seq_len)], axis=0)
    return logits, caches, h_states, c_states

def bptt_full(inputs, targets, params, caches, h_states, logits, clip_value=5.0):
    seq_len, vocab_size = logits.shape
    layers = params['layers']
    num_layers = len(layers)
    hidden_size = layers[0]['b'].shape[0] // 4

    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    dy = probs
    dy[np.arange(seq_len), targets] -= 1
    dy /= seq_len

    grads = {'layers': [{'Wxh': np.zeros_like(l['Wxh']),
                         'Whh': np.zeros_like(l['Whh']),
                         'b': np.zeros_like(l['b'])} for l in layers],
             'Wy': np.zeros_like(params['Wy']),
             'by': np.zeros_like(params['by'])}

    dh_next = [np.zeros(hidden_size) for _ in range(num_layers)]
    dc_next = [np.zeros(hidden_size) for _ in range(num_layers)]

    for t in reversed(range(seq_len)):
        # Output layer gradient
        grads['Wy'] += np.outer(h_states[-1][t], dy[t])
        grads['by'] += dy[t]
        dh = np.dot(params['Wy'], dy[t]) + dh_next[-1]

        for l_idx in reversed(range(num_layers)):
            x_t, h_prev, c_prev, i, f, g, o, c_next, Wxh, Whh, b = caches[l_idx][t]
            hidden_size = h_prev.shape[0]

            # Output gate
            do = dh * np.tanh(c_next)
            do = do * o * (1 - o)

            # Cell state
            dc = dh * o * (1 - np.tanh(c_next)**2) + dc_next[l_idx]

            # Forget gate
            df = dc * c_prev
            df = df * f * (1 - f)

            # Input gate
            di = dc * g
            di = di * i * (1 - i)

            # Candidate
            dg = dc * i
            dg = dg * (1 - g**2)

            dz = np.concatenate([di, df, dg, do])

            grads['layers'][l_idx]['Wxh'] += np.outer(x_t, dz)
            grads['layers'][l_idx]['Whh'] += np.outer(h_prev, dz)
            grads['layers'][l_idx]['b'] += dz

            dh = np.dot(Whh.T, dz)
            dc = dc * f

            dh_next[l_idx] = dh
            dc_next[l_idx] = dc

    # Gradient clipping
    for g in [grads['Wy'], grads['by']]:
        np.clip(g, -clip_value, clip_value, out=g)
    for l_idx in range(num_layers):
        for k in ['Wxh', 'Whh', 'b']:
            np.clip(grads['layers'][l_idx][k], -clip_value, clip_value, out=grads['layers'][l_idx][k])

    return grads

def sample_text(params, seed, char_to_idx, idx_to_char, n=200, temperature=1.0):
    layers = params['layers']
    num_layers = len(layers)
    hidden_size = layers[0]['b'].shape[0] // 4
    vocab_size = len(char_to_idx)

    h = [np.zeros(hidden_size) for _ in range(num_layers)]
    c = [np.zeros(hidden_size) for _ in range(num_layers)]

    # Warm-up
    for ch in seed:
        x = np.zeros(vocab_size)
        x[char_to_idx[ch]] = 1.0
        for l_idx in range(num_layers):
            h[l_idx], c[l_idx], _ = lstm_step(x, h[l_idx], c[l_idx], layers[l_idx]['Wxh'], layers[l_idx]['Whh'], layers[l_idx]['b'])
            x = h[l_idx]

    result = seed
    for _ in range(n):
        x = np.zeros(vocab_size)
        x[char_to_idx[result[-1]]] = 1.0
        for l_idx in range(num_layers):
            h[l_idx], c[l_idx], _ = lstm_step(x, h[l_idx], c[l_idx], layers[l_idx]['Wxh'], layers[l_idx]['Whh'], layers[l_idx]['b'])
            x = h[l_idx]
        logits = np.dot(h[-1], params['Wy']) + params['by']
        logits = logits / temperature
        probs = softmax(logits)
        idx = np.random.choice(vocab_size, p=probs)
        result += idx_to_char[idx]
    return result

def train():
    text = load_corpus()
    chars, char_to_idx, idx_to_char = build_vocab(text)
    data = encode(text, char_to_idx)
    vocab_size = len(chars)

    hidden_size = 128
    num_layers = 2
    seq_len = 100
    lr = 0.01
    iters = 1000

    params = init_params(vocab_size, hidden_size, num_layers)
    smooth_loss = -np.log(1.0 / vocab_size) * seq_len

    for it in range(iters):
        start = np.random.randint(0, len(data) - seq_len - 1)
        x_seq = data[start:start+seq_len]
        y_seq = data[start+1:start+seq_len+1]

        inputs = [np.zeros(vocab_size) for _ in range(seq_len)]
        for t in range(seq_len):
            inputs[t][x_seq[t]] = 1.0

        logits, caches, h_states, c_states = forward_sequence_full(inputs, params)
        loss = cross_entropy(logits, y_seq)
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        grads = bptt_full(inputs, y_seq, params, caches, h_states, logits)

        # SGD update
        for l_idx in range(num_layers):
            for k in ['Wxh', 'Whh', 'b']:
                params['layers'][l_idx][k] -= lr * grads['layers'][l_idx][k]
        params['Wy'] -= lr * grads['Wy']
        params['by'] -= lr * grads['by']

        if it % 100 == 0:
            print(f"Iteration {it}, smooth loss: {smooth_loss:.4f}")
            print(sample_text(params, "It was ", char_to_idx, idx_to_char, n=100, temperature=0.5))
            print("-" * 40)

    print("\nFinal sample (temperature=0.5):")
    print(sample_text(params, "It was ", char_to_idx, idx_to_char, n=200, temperature=0.5))

if __name__ == "__main__":
    train()
```

</details>

## What You Actually Learned

- **Recurrent Neural Networks:** You saw why vanilla RNNs struggle with long-range dependencies and why LSTMs were invented. You implemented the gating mechanism that allows gradients to flow across hundreds of time steps.
- **LSTM and GRU:** You wrote the full set of LSTM equations (forget, input, cell, output gates) and backpropagated through them manually. This demystifies what frameworks hide behind `nn.LSTM`.
- **Truncated BPTT:** You limited backpropagation to a fixed window, which is how production language models are trained on infinite streams of text.
- **Temperature sampling:** You controlled the randomness of generation by scaling logits, a technique used in every production text API (OpenAI's `temperature`, Anthropic's `temperature`).
- **Stacked RNNs:** You built a multi-layer LSTM, showing how depth increases representational power without exploding parameter counts the way width does.
