import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Generate sample data (random points)
np.random.seed(42)
X = np.random.rand(100, 2) * 10  # 100 points in a 2D space

# Step 2: Apply k-Means (let's choose k=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 3: Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"k-Means Clustering (k={k})")
plt.legend()
# plt.show()
plt.savefig("k_means_plot.png")

