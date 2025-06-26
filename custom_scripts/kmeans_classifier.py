import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your CSV
df = pd.read_csv("demo/businessman_with_headset/pywork/combined_scores.csv")
scores = df["Track 0"].values.reshape(-1, 1)

# Apply K-Means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(scores)

# Add cluster labels to the DataFrame
df["Cluster"] = labels

# Plot the scores with cluster coloring
plt.figure(figsize=(12, 5))
plt.scatter(df["Frame Index"], df["Track 0"], c=labels, cmap="coolwarm", s=10)
plt.xlabel("Frame Index")
plt.ylabel("Speaking Score")
plt.title("Unsupervised Clustering of Speaking Scores")
plt.grid(True)
plt.tight_layout()
plt.show()
