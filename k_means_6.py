import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data (Age vs Wake-up Time)
X = np.array([
    [52.,   6.  ],
    [47.,   6.3 ],
    [44.,   7.  ],
    [41.,   5.  ],
    [40.,   5.  ],
    [43.,   5.  ],
    [32.,   5.  ],
    [42.,   4.  ],
    [50.,   7.  ],
    [41.,   1.  ],
    [38.,   6.  ],
    [36.,   6.  ],
    [40.,   7.  ],
    [34.,   7.  ],
    [29.,   22.  ],
    [19.,   6.  ],
    [58.,   5.  ],
    [57.,   7.  ],
    [39.,   5.3 ],
    [42.,   0.  ],
    [26.,   6.  ],
    [57.,   5.  ],
    [27.,   8.  ],
    [53.,   6.  ],
    [25.,   7.  ],
    [41.,   3.  ],
    [32.,   6.  ],
    [41.,   4.  ],
    [50.,   6.  ],
    [41.,   5.  ],
    [39.,   5.  ],
    [28.,   8.  ],
    [58.,   5.  ],
    [54.,   5.  ],
    [47.,   4.  ],
    [34.,   8.  ],
    [67.,   3.  ],
    [17.,   22.  ],
    [31.,   6.  ],
    [30.,   8.  ],
    [53.,   7.  ],
    [43.,   5.3 ],
    [54.,   6.  ],
    [32.,   6.  ],
    [32.,   8.  ],
    [19.,   9.  ],
    [58.,   7.  ],
    [44.,   5.  ],
    [25.,   7.  ],
    [39.,   18.  ],
    [36.,   5.  ],
    [66.,   3.  ],
    [27.,   9.  ],
    [30.,   9.  ],
    [53.,   5.  ],
    [28.,   17.  ],
    [38.,   7.  ],
    [34.,   6.  ],
    [30.,   18.  ],
    [43.,   4.  ],
    [52.,   7.  ],
    [54.,   5.  ],
    [54.,   5.  ],
    [23.,   11.  ],
    [58.,   4.  ],
    [18.,   6.  ],
    [27.,   6.  ],
    [21.,   7.  ],
    [21.,   4.  ],
    [32.,   10.  ],
    [19.,   9.  ],
    [54.,   6.  ],
    [33.,   7.  ],
    [19.,   11.  ],
    [30.,   6.  ],
    [44.,   5.  ],
    [40.,   14.  ],
    [22.,   7.  ],
    [28.,   8.  ],
    [46.,   4.  ],
    [20.,   12.  ],
    [35.,   6.  ],
    [41.,   5.  ],
    [21.,   8.  ],
    [40.,   6.  ],
    [30.,   4.  ],
    [39.,   7.  ],
    [21.,   9.  ],
    [45.,   6.  ],
    [19.,   15.  ],
    [20.,   7.  ],
    [19.,   8.  ],
    [43.,   6.  ],
    [55.,   4.  ],
    [56.,   5.  ],
    [41.,   4.  ],
    [56.,   5.  ],
    [32.,   4.  ],
    [39.,   6.  ],
    [29.,   6.  ],
    [41.,   5.  ],
    [32.,   6.  ],
    [39.,   5.  ],
    [54.,   4.3 ],
    [54.,   6.  ],
    [19.,   15.  ],
    [54.,   5.3 ],
    [57.,   6.  ],
    [20.,   10.  ],
    [52.,   7.  ],
    [41.,   8.3 ],
    [26.,   6.15],
    [31.,   7.  ],
    [29.,   7.  ],
    [19.,   8.  ],
    [42.,   5.  ],
    [27.,   5.  ],
    [42.,   7.  ],
    [22.,   8.  ],
    [20.,   11.  ],
    [19.,   9.  ],
    [21.,   4.  ],
    [41.,   5.  ],
    [18.,   9.  ],
    [19.,   9.3 ],
    [34.,   8.  ],
    [22.,   7.  ],
    [20.,   15.  ],
    [28.,   6.  ],
    [32.,   8.  ],
    [19.,   5.  ],
    [16.,   6.  ],
    [45.,   6.  ],
    [41.,   17.  ],
    [28.,   10.  ],
    [20.,   8.  ],
    [19.,   6.  ],
    [22.,   7.  ],
    [22.,   12.3 ],
    [21.,   8.  ],
    [20.,   10.  ],
    [25.,   7.  ],
    [28.,   10.  ],
    [22.,   8.  ],
    [25.,   8.  ],
    [30.,   6.  ],
    [24.,   10.  ],
    [21.,   5.  ],
    [27.,   6.  ],
    [20.,   7.  ]
])

# Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run k-Means for k=2 and k=6
for k in [2, 6]:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot Clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k', label="Data Points")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")

    plt.xlabel("Standardized Age")
    plt.ylabel("Standardized Wake Up Time")
    plt.title(f"k-Means Clustering (k={k})")
    plt.legend()
    plt.grid(True)
    
    # Save plot as image
    plt.savefig(f"k_means_k{k}.png")
    plt.close()  # Close the plot to avoid overlapping figures

print("Clustering results saved as 'k_means_k2.png' and 'k_means_k6.png'.")
