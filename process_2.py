
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json

with open('cleanData2.json', 'r') as file:
    data = json.load(file)  # Correct way to load JSON from a file

# Extract relevant numerical features
# selected_features = [
#     "What is your age? (Input in years,  e.g. 19 for 19y/o)",
#     "On average, how many hours do you sleep per night? (input number only, e.g., 8 for 8 hours )",
#     "What time do you usually go to bed? (Military time, e.g., 22 for 10 PM)",
#     "What time do you usually wake up? (Military time, e.g., 6 for 6 AM)",
#     "How many times do you wake up during the night?  (e.g., number from 0-5)",
#     "How often do you exercise per week? (e.g., 0-7)",
#     "How would you rate your stress level? (Stressed have felt in past week)",
#     "How often do you use electronic devices (phone, TV, computer) before sleep?"
# ]

# selected_features = [
#     "How would you describe your sleeping environment?",
#     "How comfortable is your usual sleeping position?",
# ]

selected_features = [
    "What is your age? (Input in years,  e.g. 19 for 19y/o)",
    "  What time do you usually wake up? (Military time, e.g., 6 for 6 AM)  ",
]

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Ensure relevant columns exist before selecting them
df_selected = df[selected_features].dropna().astype(float)

# Convert DataFrame to NumPy array for clustering
X = df_selected.to_numpy()

# Apply k-Means clustering with k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)

# Retrieve labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Show the first few clustered data points and centroids

numberOfPeoplesData = 150
X[:numberOfPeoplesData], labels[:5], centroids


print('X[:5] ==>> ', X[:numberOfPeoplesData]) #  First 5 people's data
print('labels[:5] ==>> ', labels[:150])   #    Their cluster assignments
print('centroids ==>> ', centroids) #  The "average" values of each cluster



# ==============================================
# What Does This Mean?
# Cluster 0: Older people (avg age 47.76), sleeping ~6.4 hours, going to bed around 20:40 (8:40 PM).
# Cluster 1: Younger group (avg age 26.51), sleeping 7 hours, going to bed at 00:27 (12:27 AM).
# Cluster 2: Youngest (avg age 25.31), very late sleepers (~04:50 AM wake-up time).


# Cluster 0: [47.76, 6.41, 20.67, 7.00, 1.70, 2.16, 3.94, 3.33]
# Cluster 1: [26.51, 7.00, 24.27, 7.45, 1.53, 1.73, 4.27, 3.31]
# Cluster 2: [25.31, 7.14,  4.83, 9.86, 1.31, 1.53, 5.06, 3.14]
# ==============================================
