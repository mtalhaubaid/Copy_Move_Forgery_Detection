from sklearn.cluster import AgglomerativeClustering

# Sample data
data = [[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]]

# Create a model with a given number of clusters
# For instance, we're setting it to 2 clusters here
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

# Fit the model and predict clusters for your data
cluster_id = cluster.fit_predict(data)

print(cluster_id)  # This will print the cluster IDs for each data point
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linked = linkage(data, 'ward')

dendrogram(linked)
plt.show()
