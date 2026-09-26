#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# %%
iris  = datasets.load_iris()
# %%
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# %%
iris_data
# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# %%
X_scaled  = scaler.fit_transform(iris_data)
# %%
X_scaled
# %%
X_scaled.shape
# %%
## USE PCA to reduce the dimensions of the data
from sklearn.decomposition import PCA
# %%
pca = PCA(n_components=2)
# %%
pca_scaled  = pca.fit_transform(X_scaled)
pca_scaled.shape
# %%
plt.scatter(pca_scaled[:,0],pca_scaled[:,1])
# %%
## Aggolomerative Clustering
import scipy.cluster.hierarchy as sc
plt.figure(figsize=(20,7))
plt.title("Dendrograms")

sc.dendrogram(sc.linkage(pca_scaled, method='ward'))
plt.title("Dendrograms")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distances")
# %%
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
cluster.fit(pca_scaled)
# %%
cluster.labels_
# %%
plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=cluster.labels_)
# %%
from sklearn.metrics import silhouette_score
silhouette_coefficients = []
# %%
for k in range(2,11):
    cluster = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    cluster.fit(pca_scaled)
    score = silhouette_score(pca_scaled, cluster.labels_)
    silhouette_coefficients.append(score)
# %%
plt.plot(range(2,11), silhouette_coefficients)
plt.xticks(range(2,11))
plt.title("Silhouette Coefficients")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
# %%
