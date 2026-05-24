# Practice: RAG QA System

**Phase:** PHASE-04-llm-engineering  
**Subjects Required:** 86 — RAG: Retrieval, 87 — Vector Databases, 88 — Chunking Strategies  
**Estimated Time:** 270 minutes  
**Difficulty:** Intermediate

## Industry Context

You are the founding engineer at a Series A health-tech startup. Your product team wants to let doctors query a knowledge base of 50,000 clinical guidelines without sending protected data to a third-party LLM API. You must build an on-premise retrieval-augmented generation (RAG) QA system that runs entirely inside the hospital's VPC. Latency must stay under 2 seconds per query, and answers must cite the exact guideline chunks they were derived from.

## The Problem

Build a minimal but production-grade RAG pipeline that:

1. **Chunks** a collection of plain-text documents using a fixed-size sliding-window strategy with overlap.
2. **Embeds** each chunk into a dense vector using a sentence-transformer model.
3. **Stores** vectors in an in-memory FAISS index with metadata mapping vectors back to chunk text and source document.
4. **Retrieves** the top-k most relevant chunks for a user query.
5. **Generates** a grounded answer by concatenating the retrieved chunks into a prompt template and running inference with a small local LLM (simulated here with a deterministic function).
6. **Cites** the source documents used in the answer.

You may use `numpy`, `faiss-cpu`, and `sentence-transformers`. Do not use LangChain, LlamaIndex, or any other high-level RAG framework. The goal is to understand the glue, not the abstraction.

## Constraints

- Must run on a single CPU core in under 5 seconds end-to-end for a single query (excluding model download time).
- Chunk size: 256 tokens, overlap: 32 tokens. Tokenization can be approximate (split on whitespace).
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dim vectors).
- FAISS index: `IndexFlatIP` (exact inner-product search; vectors are L2-normalized so IP == cosine similarity).
- Do not persist data to disk. Everything lives in memory for this exercise.
- The answer generation step must be deterministic: given the same retrieved chunks, it always produces the same output. Use a simple rule-based "LLM" function provided in the starter code.

## Starter Code

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
CHUNK_SIZE = 256          # tokens
OVERLAP = 32              # tokens
TOP_K = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# 2. DOCUMENTS (simulated clinical guidelines)
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "guideline_001": (
        "Hypertension management in adults. First-line therapy for uncomplicated "
        "hypertension is a thiazide diuretic, calcium channel blocker, ACE inhibitor, "
        "or ARB. Monitor potassium when prescribing thiazides. Target blood pressure "
        "is less than 130 over 80 mmHg for most adults. For patients over 65, a target "
        "of less than 130 over 80 is also recommended if tolerated. Lifestyle modifications "
        "include sodium restriction, weight loss, and regular aerobic exercise. "
        "Resistant hypertension is defined as failure to achieve target despite three "
        "drugs including a diuretic. "
    ) * 5,  # repeat to make it long enough to chunk
    "guideline_002": (
        "Diabetes mellitus type 2 screening and treatment. Screen adults aged 35 to 70 "
        "who are overweight or obese. First-line pharmacotherapy is metformin. Monitor "
        "hemoglobin A1c every three months until stable, then every six months. Target "
        "A1c is less than 7 percent for most non-pregnant adults. Consider insulin if "
        "A1c remains above target despite two oral agents. "
    ) * 5,
    "guideline_003": (
        "Statin therapy for primary prevention. Prescribe moderate-intensity statin for "
        "adults aged 40 to 75 with one or more cardiovascular risk factors and a 10-year "
        "ASCVD risk of 10 percent or greater. High-intensity statin is indicated for "
        "LDL cholesterol 190 mg per dL or higher. Monitor liver enzymes at baseline and "
        "then as clinically indicated. "
    ) * 5,
}

# ---------------------------------------------------------------------------
# 3. TOKENIZATION (approximate: whitespace split)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return text.split()

# ---------------------------------------------------------------------------
# 4. CHUNKING
# ---------------------------------------------------------------------------
def chunk_document(doc_id: str, text: str, chunk_size: int, overlap: int) -> List[dict]:
    """
    TODO: Split `text` into overlapping chunks.
    Each chunk dict must contain:
      - "doc_id": str
      - "chunk_index": int
      - "text": str   (the chunk text, joined with spaces)
    Return the list of chunks in order.
    """
    raise NotImplementedError("Implement the sliding-window chunker.")

# ---------------------------------------------------------------------------
# 5. EMBEDDING
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        TODO: Encode a list of texts into a (N, D) float32 array.
        L2-normalize each vector so that inner-product search equals cosine similarity.
        """
        raise NotImplementedError("Implement encode with L2 normalization.")

# ---------------------------------------------------------------------------
# 6. VECTOR STORE
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, dim: int):
        """
        TODO: Initialize a FAISS IndexFlatIP and any metadata structures you need.
        """
        raise NotImplementedError("Initialize the FAISS index.")

    def add(self, vectors: np.ndarray, metadatas: List[dict]):
        """
        TODO: Add vectors and their metadata to the index.
        """
        raise NotImplementedError("Add vectors to the store.")

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, List[dict]]:
        """
        TODO: Search the index for the k nearest neighbors.
        Return (distances, metadatas) where metadatas is a list of the top-k metadata dicts.
        """
        raise NotImplementedError("Search the store.")

# ---------------------------------------------------------------------------
# 7. RAG PIPELINE
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def ingest(self, documents: dict):
        """
        TODO: Chunk all documents, embed the chunks, and add them to the vector store.
        """
        raise NotImplementedError("Ingest documents into the pipeline.")

    def query(self, question: str) -> dict:
        """
        TODO:
          1. Embed the question.
          2. Retrieve top_k chunks.
          3. Build a prompt from the chunks.
          4. Call generate_answer(prompt, chunks).
          5. Return {"answer": str, "sources": List[str], "chunks": List[str]}
             where sources is the deduplicated list of doc_ids used.
        """
        raise NotImplementedError("Run the RAG query.")

# ---------------------------------------------------------------------------
# 8. DETERMINISTIC "LLM" (simulated)
# ---------------------------------------------------------------------------
def generate_answer(prompt: str, chunks: List[dict]) -> str:
    """
    Simulates a grounded LLM. It simply returns the concatenated chunk texts.
    In production this would be a call to a local model like Llama-2-7B or Mistral-7B.
    """
    return "Answer derived from the following context:\n\n" + "\n---\n".join(
        c["text"] for c in chunks
    )

# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embedder = Embedder(MODEL_NAME)
    store = VectorStore(embedder.dim)
    rag = RAGPipeline(embedder, store, TOP_K)
    rag.ingest(DOCUMENTS)

    question = "What is the first-line therapy for hypertension and how often should we monitor?"
    result = rag.query(question)

    print("Answer:\n", result["answer"])
    print("\nSources:", result["sources"])
    print("\nChunks used:", len(result["chunks"]))
```

## Evaluation Criteria

1. **Correctness:** The pipeline runs end-to-end without errors and produces a non-empty answer with at least one source citation.
2. **Chunking:** Documents are split into overlapping chunks of the correct size. The overlap preserves context across chunk boundaries.
3. **Embedding:** Vectors are L2-normalized before insertion so that FAISS `IndexFlatIP` returns cosine-similarity-ranked results.
4. **Retrieval:** The top-k chunks returned for the hypertension query include chunks from `guideline_001`.
5. **Citation:** The `sources` list contains the unique `doc_id`s of the retrieved chunks, not the chunk indices.
6. **Efficiency:** Ingestion + query completes in under 5 seconds on CPU after model download.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
CHUNK_SIZE = 256
OVERLAP = 32
TOP_K = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# 2. DOCUMENTS
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "guideline_001": (
        "Hypertension management in adults. First-line therapy for uncomplicated "
        "hypertension is a thiazide diuretic, calcium channel blocker, ACE inhibitor, "
        "or ARB. Monitor potassium when prescribing thiazides. Target blood pressure "
        "is less than 130 over 80 mmHg for most adults. For patients over 65, a target "
        "of less than 130 over 80 is also recommended if tolerated. Lifestyle modifications "
        "include sodium restriction, weight loss, and regular aerobic exercise. "
        "Resistant hypertension is defined as failure to achieve target despite three "
        "drugs including a diuretic. "
    ) * 5,
    "guideline_002": (
        "Diabetes mellitus type 2 screening and treatment. Screen adults aged 35 to 70 "
        "who are overweight or obese. First-line pharmacotherapy is metformin. Monitor "
        "hemoglobin A1c every three months until stable, then every six months. Target "
        "A1c is less than 7 percent for most non-pregnant adults. Consider insulin if "
        "A1c remains above target despite two oral agents. "
    ) * 5,
    "guideline_003": (
        "Statin therapy for primary prevention. Prescribe moderate-intensity statin for "
        "adults aged 40 to 75 with one or more cardiovascular risk factors and a 10-year "
        "ASCVD risk of 10 percent or greater. High-intensity statin is indicated for "
        "LDL cholesterol 190 mg per dL or higher. Monitor liver enzymes at baseline and "
        "then as clinically indicated. "
    ) * 5,
}

# ---------------------------------------------------------------------------
# 3. TOKENIZATION
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return text.split()

# ---------------------------------------------------------------------------
# 4. CHUNKING
# ---------------------------------------------------------------------------
def chunk_document(doc_id: str, text: str, chunk_size: int, overlap: int) -> List[dict]:
    tokens = tokenize(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": " ".join(chunk_tokens),
        })
        start += chunk_size - overlap
        idx += 1
    return chunks

# ---------------------------------------------------------------------------
# 5. EMBEDDING
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = vectors.astype(np.float32)
        # L2-normalize so inner product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        return vectors

# ---------------------------------------------------------------------------
# 6. VECTOR STORE
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[dict] = []

    def add(self, vectors: np.ndarray, metadatas: List[dict]):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, List[dict]]:
        # FAISS expects (n_queries, dim)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        top_metadatas = [self.metadatas[i] for i in indices[0]]
        return distances[0], top_metadatas

# ---------------------------------------------------------------------------
# 7. RAG PIPELINE
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def ingest(self, documents: dict):
        all_chunks = []
        for doc_id, text in documents.items():
            all_chunks.extend(chunk_document(doc_id, text, CHUNK_SIZE, OVERLAP))
        texts = [c["text"] for c in all_chunks]
        vectors = self.embedder.encode(texts)
        self.store.add(vectors, all_chunks)

    def query(self, question: str) -> dict:
        q_vector = self.embedder.encode([question])[0]
        _, top_chunks = self.store.search(q_vector, self.top_k)
        prompt = (
            "You are a clinical assistant. Answer the question using only the context below.\n\n"
            "Context:\n" + "\n---\n".join(c["text"] for c in top_chunks) + "\n\n"
            "Question: " + question + "\nAnswer:"
        )
        answer = generate_answer(prompt, top_chunks)
        sources = list({c["doc_id"] for c in top_chunks})
        return {
            "answer": answer,
            "sources": sources,
            "chunks": [c["text"] for c in top_chunks],
        }

# ---------------------------------------------------------------------------
# 8. DETERMINISTIC "LLM"
# ---------------------------------------------------------------------------
def generate_answer(prompt: str, chunks: List[dict]) -> str:
    return "Answer derived from the following context:\n\n" + "\n---\n".join(
        c["text"] for c in chunks
    )

# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embedder = Embedder(MODEL_NAME)
    store = VectorStore(embedder.dim)
    rag = RAGPipeline(embedder, store, TOP_K)
    rag.ingest(DOCUMENTS)

    question = "What is the first-line therapy for hypertension and how often should we monitor?"
    result = rag.query(question)

    print("Answer:\n", result["answer"])
    print("\nSources:", result["sources"])
    print("\nChunks used:", len(result["chunks"]))
```

</details>

## What You Actually Learned

- **Chunking Strategies:** You implemented a fixed-size sliding-window chunker with overlap. This preserves semantic continuity at chunk boundaries and is the default strategy in most production RAG systems before moving to more advanced semantic chunking.
- **Vector Databases:** You built an in-memory FAISS index, normalized embeddings for cosine-similarity search, and maintained metadata to map vectors back to source documents. This is the exact pattern used inside Chroma, Pinecone, and Weaviate.
- **RAG: Retrieval:** You wired retrieval into a generation loop, constructed a grounded prompt, and returned citations. This is the core architecture of every modern enterprise knowledge-base product.
