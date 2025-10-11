# Example 2: Clustering gene expression samples
# This example demonstrates how to cluster gene expression data using K-Means clustering.

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fake dataset: 100 samples, 2 genes
X = np.random.rand(100, 2)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("K-Means Clustering of Gene Expression Data")
plt.xlabel("Gene 1 Expression")
plt.ylabel("Gene 2 Expression")
plt.show()

