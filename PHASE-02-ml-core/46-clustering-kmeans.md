# Clustering with K-Means

**Phase:** PHASE-02-ml-core  
**Prerequisites:** 3 (Dot Product), 34 (ML Workflow Overview)  
**Estimated Time:** 50 minutes

## Why am I learning this?

Most real-world data arrives without labels. A product team asks you to "group similar users" or "find patterns in sensor readings," and they do not have the budget or time to annotate thousands of points. K-Means is the algorithm you reach for first in these situations—not because it is perfect, but because it is fast, interpretable, and gives you a baseline in under ten lines of code.

You will rarely ship K-Means as the final model in production, but you will use it constantly for exploratory analysis, pre-clustering before supervised learning, and sanity-checking whether your data has any structure at all. Understanding its mechanics also prepares you for more advanced clustering methods (DBSCAN, Gaussian Mixture Models) and for vector-database retrieval, where distance-based partitioning is the core operation.

## Where will I be using it?

- **Customer Segmentation:** Grouping users by purchase behavior for targeted marketing campaigns.
- **Anomaly Detection:** Points far from any centroid are flagged as outliers.
- **Image Compression:** Reducing the color palette of an image to $k$ representative colors.
- **Document Clustering:** Grouping news articles or support tickets by TF-IDF similarity.
- **Pre-processing for Supervised Learning:** Using cluster labels as additional features.

## Resources

- [scikit-learn: Clustering — K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means) — Official documentation covering Lloyd's algorithm, inertia, and k-means++ initialization.
- [scikit-learn: KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — The API you will actually call.
- [Wikipedia: K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) — Historical context, algorithm variants, and convergence properties.
- [arXiv: K+ Means — An Enhancement Over K-Means](https://arxiv.org/abs/1706.02949) — Discusses the classic algorithm and challenges in choosing $k$.
- [arXiv: Scalable Kernel Clustering — Approximate Kernel k-means](https://arxiv.org/abs/1402.3849) — Extension to non-linear cluster boundaries via kernel methods.

## Appendix

### Notation

- $X = \{x_1, x_2, \dots, x_n\}$: Dataset with $n$ samples, each $x_i \in \mathbb{R}^d$.
- $k$: Number of clusters (hyperparameter).
- $C_j$: The $j$-th cluster, a subset of $X$.
- $\mu_j \in \mathbb{R}^d$: Centroid (mean) of cluster $C_j$.
- $\text{inertia} = \sum_{i=1}^{n} \min_{\mu_j \in C} \|x_i - \mu_j\|^2$: Within-cluster sum-of-squares (WCSS).

### Algorithm (Lloyd's)

1. **Initialize** $k$ centroids $\mu_1, \dots, \mu_k$ (commonly via k-means++).
2. **Assign** each sample $x_i$ to the nearest centroid:
   $$C_j = \{x_i : \|x_i - \mu_j\|^2 \leq \|x_i - \mu_l\|^2 \; \forall \, l\}$$
3. **Update** each centroid to the mean of its assigned samples:
   $$\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$
4. **Repeat** steps 2–3 until centroids move less than a tolerance or a max iteration count is reached.

### Common Pitfalls

- **Choosing $k$:** There is no universal rule. The elbow method (plotting inertia vs. $k$) is a heuristic, not a guarantee. Silhouette analysis is more robust but slower.
- **Sensitivity to initialization:** Random initialization can converge to poor local minima. Always use `init='k-means++'` (the default in scikit-learn).
- **Spherical assumption:** K-Means assumes clusters are convex and isotropic. It fails on elongated or irregularly shaped clusters.
- **Curse of dimensionality:** Euclidean distances become less meaningful in very high-dimensional spaces. Run PCA first if $d$ is large.
- **Scaling matters:** Features with larger scales dominate the distance metric. Standardize features (`StandardScaler`) before clustering.

### Further Reading

- [scikit-learn: Selecting the number of clusters with silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) — Practical guide to choosing $k$.
- [Distill.pub: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — Cautionary notes on visualizing clusters in low dimensions.
