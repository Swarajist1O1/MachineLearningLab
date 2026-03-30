import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(load_iris().data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
c = kmeans.fit_predict(X)

print("Silhouette Score:", silhouette_score(X, c))

plt.scatter(X[:,0], X[:,1], c=c)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='X', s=200)
plt.title("K-Means (k=3)")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()