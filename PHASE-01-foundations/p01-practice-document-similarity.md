# Practice: Document Similarity Engine

**Phase:** PHASE-01-foundations  
**Subjects Required:** 01-vectors, 02-vector-operations, 03-dot-product  
**Estimated Time:** 120 minutes  
**Difficulty:** Beginner

## Industry Context

You are the first ML engineer at a legal-tech startup. The product team wants semantic search across 100,000 contracts so lawyers can find similar clauses quickly. You cannot afford OpenAI API costs at scale. The CEO asks you to prototype an in-house retrieval system using classical methods.

Your task: build a TF-IDF + cosine similarity search engine from scratch. This is not a toy problem—this is how Elasticsearch, Lucene, and early versions of semantic search worked before embeddings became cheap.

## The Problem

Implement a `DocumentSearcher` class that:

1. Takes a corpus of raw text documents.
2. Builds a TF-IDF vector for each document using only NumPy.
3. Given a query string, returns the top-k most similar documents using cosine similarity.

You must implement TF-IDF from scratch. Do not use scikit-learn, NLTK, or any library that computes TF-IDF for you. Tokenization can be naive (split on whitespace and lowercase).

## Constraints

- Use only **NumPy** and Python standard library.
- Must handle a corpus of at least 1,000 documents in under 2 seconds on a single CPU.
- Must handle queries that contain words not seen in the corpus (gracefully return zero similarity for unknown terms).
- The TF-IDF vectors must be normalized so cosine similarity is just the dot product.

## Starter Code

```python
import numpy as np
from typing import List, Tuple

class DocumentSearcher:
    def __init__(self):
        self.vocab = {}          # word -> index
        self.idf = None          # shape: (vocab_size,)
        self.doc_matrix = None   # shape: (num_docs, vocab_size)
        self.documents = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def fit(self, documents: List[str]):
        self.documents = documents
        # TODO: Build vocabulary from all documents
        # TODO: Compute IDF for each term
        # TODO: Build document-term matrix with TF-IDF values
        # TODO: Normalize each document vector to unit length
        pass

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        # TODO: Tokenize query
        # TODO: Convert query to TF-IDF vector (using the same IDF values)
        # TODO: Normalize query vector
        # TODO: Compute cosine similarity via dot product with all documents
        # TODO: Return top-k (doc_index, similarity_score)
        pass


# Test corpus
CORPUS = [
    "machine learning is fascinating",
    "deep learning requires large datasets",
    "natural language processing is a subset of machine learning",
    "computer vision uses convolutional neural networks",
    "neural networks are the foundation of deep learning",
    "data science combines statistics and machine learning",
    "reinforcement learning trains agents through rewards",
    "supervised learning uses labeled data",
]

if __name__ == "__main__":
    searcher = DocumentSearcher()
    searcher.fit(CORPUS)
    results = searcher.search("machine learning", k=3)
    for idx, score in results:
        print(f"Doc {idx} (score={score:.4f}): {CORPUS[idx]}")
```

## Evaluation Criteria

1. **Correctness:** Searching for "machine learning" should return documents containing those words with the highest scores.
2. **Edge handling:** Query "quantum computing" (not in corpus) should return empty or zero scores without crashing.
3. **Efficiency:** `search()` should use matrix-vector multiplication, not a Python loop over documents.
4. **Normalization:** After normalization, the similarity between identical documents should be `1.0`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
from typing import List, Tuple
import math

class DocumentSearcher:
    def __init__(self):
        self.vocab = {}          # word -> index
        self.idf = None          # shape: (vocab_size,)
        self.doc_matrix = None   # shape: (num_docs, vocab_size)
        self.documents = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def fit(self, documents: List[str]):
        self.documents = documents
        tokenized_docs = [self._tokenize(d) for d in documents]

        # Build vocabulary
        unique_words = set()
        for tokens in tokenized_docs:
            unique_words.update(tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(unique_words))}
        vocab_size = len(self.vocab)
        num_docs = len(documents)

        # Compute raw term frequencies
        tf = np.zeros((num_docs, vocab_size))
        for i, tokens in enumerate(tokenized_docs):
            for token in tokens:
                if token in self.vocab:
                    tf[i, self.vocab[token]] += 1

        # Normalize TF (term frequency / total terms in document)
        doc_lengths = tf.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1  # avoid division by zero
        tf = tf / doc_lengths

        # Compute IDF: log(N / df) where df is number of docs containing the term
        df = (tf > 0).sum(axis=0)  # document frequency for each term
        self.idf = np.log(num_docs / (df + 1e-10))

        # TF-IDF
        self.doc_matrix = tf * self.idf

        # L2-normalize each document vector so cosine similarity = dot product
        norms = np.linalg.norm(self.doc_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.doc_matrix = self.doc_matrix / norms

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        query_vec = np.zeros(len(self.vocab))

        # Build raw count vector for query
        for token in tokens:
            if token in self.vocab:
                query_vec[self.vocab[token]] += 1

        # Apply IDF (same as documents)
        query_vec = query_vec * self.idf

        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        # Cosine similarity via dot product (because both are normalized)
        # This is a matrix-vector multiplication: (num_docs, vocab_size) @ (vocab_size,)
        similarities = self.doc_matrix @ query_vec

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


# Test corpus
CORPUS = [
    "machine learning is fascinating",
    "deep learning requires large datasets",
    "natural language processing is a subset of machine learning",
    "computer vision uses convolutional neural networks",
    "neural networks are the foundation of deep learning",
    "data science combines statistics and machine learning",
    "reinforcement learning trains agents through rewards",
    "supervised learning uses labeled data",
]

if __name__ == "__main__":
    searcher = DocumentSearcher()
    searcher.fit(CORPUS)
    results = searcher.search("machine learning", k=3)
    for idx, score in results:
        print(f"Doc {idx} (score={score:.4f}): {CORPUS[idx]}")
```

</details>

## What You Actually Learned

- **Vectors:** Each document became a high-dimensional vector in vocabulary space. This is the core representation in classical NLP.
- **Vector Operations:** You added counts, scaled by IDF, and normalized—all vector operations.
- **Dot Product:** Because you normalized both query and documents, the entire search reduced to a single matrix-vector dot product. This is why cosine similarity is fast at scale.
- **Industry Reality:** Before transformers, this is how search worked. TF-IDF + cosine similarity powers Elasticsearch, Solr, and many production retrieval systems today.
