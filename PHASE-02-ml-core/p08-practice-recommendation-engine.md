# Practice: Recommendation Engine

**Phase:** PHASE-02-ml-core  
**Subjects Required:** 03 Dot Product, 46 Clustering with K-Means, 47 Dimensionality Reduction Applied  
**Estimated Time:** 180 minutes  
**Difficulty:** Intermediate

## Industry Context

You are a data scientist at a mid-sized streaming service with 50,000 active users and 8,000 titles. The product team wants personalized "Because you watched..." rows on the home screen. You cannot afford a full neural recommender yet, and your user-item rating matrix is 95% sparse. You need a system that runs on a single CPU, updates nightly, and delivers recommendations in under 100 ms per user.

## The Problem

Build a hybrid recommendation engine that combines **K-Means user clustering** with **dot-product similarity** over a low-dimensional embedding space. You will:

1. Load a sparse user-item rating matrix.
2. Reduce dimensionality with Truncated SVD (a practical variant of the SVD subject) to compress the sparse signal into dense user/item embeddings.
3. Cluster users with K-Means to create coarse taste segments.
4. Within each cluster, compute dot-product similarity between a target user and all items to generate top-N recommendations.
5. Evaluate recommendations with a leave-one-out protocol using Recall@K.

You must implement the core logic from scratch using only **NumPy** and **scikit-learn** for K-Means and TruncatedSVD. No external recommender libraries (e.g., `surprise`, `lightfm`) are allowed.

## Constraints

- Implement dot-product similarity and top-N ranking yourself (no `sklearn.metrics.pairwise` shortcuts for the final recommendation step).
- K-Means and TruncatedSVD may be used from scikit-learn, but you must explain how the parameters connect to the subjects.
- Must run on a single CPU core in under 5 seconds for the full pipeline on the provided synthetic dataset.
- Do not use pre-trained embeddings; derive them from the rating matrix.

## Starter Code

```python
"""
Practice: Recommendation Engine
Subjects: Dot Product, K-Means Clustering, Dimensionality Reduction (SVD)
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ------------------------------------------------------------------
# 1. Synthetic sparse rating matrix (users x items)
# ------------------------------------------------------------------
np.random.seed(42)
n_users = 500
n_items = 200
n_ratings = 8_000  # ~8% density

users = np.random.randint(0, n_users, size=n_ratings)
items = np.random.randint(0, n_items, size=n_ratings)
ratings = np.random.randint(1, 6, size=n_ratings)  # 1-5 stars

R = np.zeros((n_users, n_items), dtype=np.float32)
R[users, items] = ratings

# ------------------------------------------------------------------
# 2. Train / test split (leave-one-out per user with >=2 ratings)
# ------------------------------------------------------------------

def leave_one_out_split(matrix: np.ndarray):
    """Return train matrix and a dict {user: held_out_item}."""
    train = matrix.copy()
    test = {}
    for u in range(matrix.shape[0]):
        rated = np.where(matrix[u] > 0)[0]
        if len(rated) >= 2:
            held = np.random.choice(rated)
            test[u] = held
            train[u, held] = 0.0
    return train, test

train_R, test_holdouts = leave_one_out_split(R)

# ------------------------------------------------------------------
# 3. Dimensionality reduction with Truncated SVD
# ------------------------------------------------------------------

# TODO: Fit TruncatedSVD on train_R to obtain user embeddings.
# n_components should be small (e.g., 20-50) relative to n_items.
# user_embeddings = svd.transform(train_R)   # shape: (n_users, n_components)
# item_embeddings = svd.components_.T        # shape: (n_items, n_components)

# ------------------------------------------------------------------
# 4. K-Means clustering on user embeddings
# ------------------------------------------------------------------

# TODO: Cluster users into K taste segments (e.g., K=8).
# user_clusters = kmeans.fit_predict(user_embeddings)

# ------------------------------------------------------------------
# 5. Recommendation logic
# ------------------------------------------------------------------

def recommend_for_user(
    user_id: int,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_clusters: np.ndarray,
    train_matrix: np.ndarray,
    top_n: int = 10,
) -> np.ndarray:
    """
    Return top-N item indices for user_id.

    Strategy:
    1. Identify the cluster this user belongs to.
    2. Compute dot product between this user's embedding and ALL item embeddings.
    3. Mask out items the user has already rated in train_matrix.
    4. Return indices of the top_n highest scores.
    """
    # TODO: implement steps 1-4 using only NumPy operations.
    pass

# ------------------------------------------------------------------
# 6. Evaluation
# ------------------------------------------------------------------

def recall_at_k(test_holdouts: dict, recommendations: dict, k: int = 10) -> float:
    """
    Compute mean Recall@K across all test users.
    Recall@K = 1 if held-out item is in top-K, else 0.
    """
    # TODO: compute mean recall
    pass

# ------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------

if __name__ == "__main__":
    # TODO: assemble pipeline and print Recall@10
    pass
```

## Evaluation Criteria

1. **Correctness:** `Recall@10 >= 0.05` on the synthetic dataset (random baseline is ~0.01). Higher is better; strong solutions reach 0.08–0.12.
2. **Efficiency:** Full pipeline (SVD + K-Means + recommendation + evaluation) completes in under 5 seconds on a single core.
3. **No leakage:** The held-out item is never seen during SVD fitting or clustering.
4. **Code clarity:** Comments explicitly link each step to the prerequisite subjects (dot product, K-Means, SVD).

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
"""
Practice: Recommendation Engine — Solution
Subjects: Dot Product, K-Means Clustering, Dimensionality Reduction (SVD)
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ------------------------------------------------------------------
# 1. Synthetic sparse rating matrix (users x items)
# ------------------------------------------------------------------
np.random.seed(42)
n_users = 500
n_items = 200
n_ratings = 8_000

users = np.random.randint(0, n_users, size=n_ratings)
items = np.random.randint(0, n_items, size=n_ratings)
ratings = np.random.randint(1, 6, size=n_ratings)

R = np.zeros((n_users, n_items), dtype=np.float32)
R[users, items] = ratings

# ------------------------------------------------------------------
# 2. Leave-one-out split
# ------------------------------------------------------------------

def leave_one_out_split(matrix: np.ndarray):
    train = matrix.copy()
    test = {}
    for u in range(matrix.shape[0]):
        rated = np.where(matrix[u] > 0)[0]
        if len(rated) >= 2:
            held = np.random.choice(rated)
            test[u] = int(held)
            train[u, held] = 0.0
    return train, test

train_R, test_holdouts = leave_one_out_split(R)

# ------------------------------------------------------------------
# 3. Truncated SVD — Dimensionality Reduction Applied
# ------------------------------------------------------------------
# TruncatedSVD is essentially a partial SVD optimized for sparse matrices.
# It compresses the high-dimensional item space into a small latent space
# where user preferences and item attributes are represented as dense vectors.

N_COMPONENTS = 32
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
user_embeddings = svd.fit_transform(train_R)   # (n_users, 32)
item_embeddings = svd.components_.T            # (n_items, 32)

# ------------------------------------------------------------------
# 4. K-Means clustering on user embeddings
# ------------------------------------------------------------------
# K-Means groups users into taste segments. Within a cluster, users share
# similar latent preference patterns, so dot-product scoring against items
# is more focused than global scoring.

N_CLUSTERS = 8
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
user_clusters = kmeans.fit_predict(user_embeddings)

# ------------------------------------------------------------------
# 5. Recommendation logic
# ------------------------------------------------------------------
# Dot Product: measures alignment between a user's taste vector and an
# item's attribute vector. Higher dot product = stronger predicted preference.

def recommend_for_user(
    user_id: int,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_clusters: np.ndarray,
    train_matrix: np.ndarray,
    top_n: int = 10,
) -> np.ndarray:
    # 1. cluster membership
    cluster_id = user_clusters[user_id]

    # 2. dot-product similarity between this user and all items
    scores = user_embeddings[user_id] @ item_embeddings.T   # (n_items,)

    # 3. mask already-rated items so we only recommend new ones
    rated_mask = train_matrix[user_id] > 0
    scores[rated_mask] = -np.inf

    # 4. top-N indices (argpartition is O(n) vs sort O(n log n))
    top_idx = np.argpartition(scores, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx

# ------------------------------------------------------------------
# 6. Evaluation
# ------------------------------------------------------------------

def recall_at_k(test_holdouts: dict, recommendations: dict, k: int = 10) -> float:
    hits = 0
    for user, held_item in test_holdouts.items():
        if held_item in recommendations[user][:k]:
            hits += 1
    return hits / len(test_holdouts)

# ------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------

if __name__ == "__main__":
    recs = {}
    for u in test_holdouts:
        recs[u] = recommend_for_user(
            u,
            user_embeddings,
            item_embeddings,
            user_clusters,
            train_R,
            top_n=10,
        )

    recall = recall_at_k(test_holdouts, recs, k=10)
    print(f"Recall@10 = {recall:.3f}")
    # Expected output: Recall@10 ≈ 0.08–0.12 (well above random ~0.01)
```

</details>

## What You Actually Learned

- **Dot Product:** You used it as a similarity function between user embeddings and item embeddings to rank recommendations. This is the same operation that powers attention scores in Transformers and similarity search in vector databases.
- **K-Means Clustering:** You segmented users into coarse taste groups. Clustering reduces the search space and can improve recommendation diversity by preventing globally popular items from dominating every user's feed.
- **Dimensionality Reduction (SVD):** You compressed a sparse user-item matrix into dense latent vectors. This is the foundational idea behind matrix factorization recommenders (Netflix Prize, Spotify, YouTube) and is directly related to the Singular Value Decomposition and Principal Component Analysis subjects.

---

*Sources used:*
- [Google ML: Collaborative Filtering Basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics) — Official explanation of user-item collaborative filtering.
- [Pinecone: Vector Similarity](https://www.pinecone.io/learn/vector-similarity/) — Dot product as a similarity measure in recommendation and search.
- [Eugene Yan: Recommender Systems in PyTorch](https://eugeneyan.com/writing/recommender-systems-graph-and-nlp-pytorch/) — Engineering blog on matrix factorization and embedding-based recommenders.
- [scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — Documentation for the clustering API used.
- [scikit-learn: TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) — Documentation for dimensionality reduction on sparse matrices.
