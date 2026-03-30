import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

X = StandardScaler().fit_transform(load_iris().data)

Z = linkage(X, method='ward')
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Dendrogram"); plt.show()

c = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=c)
plt.title("Clustering (k=3)"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()