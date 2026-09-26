#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# %%
X,y = make_blobs(n_samples = 1000, centers=3, n_features = 2, random_state = 23)

# %%
X.shape
# %%
plt.scatter(X[:,0],X[:,1])
# %%
from sklearn.model_selection import train_test_split
# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=23)
# %%
from sklearn.cluster import KMeans
# %%
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
# %%
wcss
# %%
## Plotting Elbow curve 
plt.plot(range(1,11), wcss)
plt.xticks(range(1,11))
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# %%
kmeans = KMeans(n_clusters=3, init='k-means++')
y_labels = kmeans.fit_predict(X_train)
# %%
y_test_label = kmeans.predict(X_test)
plt.scatter(X_train[:,0],X_train[:,1],c=y_labels)


# %%
plt.scatter(X_test[:,0],X_test[:,1],c=y_test_label)

# %%
## Knee Locator
!pip install kneed 
# %%
from kneed import KneeLocator
knee = KneeLocator(range(1,11), wcss, curve='convex',
                    direction='decreasing')
## As curve is decreasing set curve  = convex and direction = decreasing
# %%
knee.elbow
# %%
## Performance Validation
from sklearn.metrics import silhouette_score

# %%
silhoutte_coefficients = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train)
    score = silhouette_score(X_train, kmeans.labels_)
    silhoutte_coefficients.append(score)
# %%
silhoutte_coefficients
# %%
plt.plot(range(2,11), silhoutte_coefficients)
plt.xticks(range(2,11))
plt.title('Silhouette Coefficient Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()
# %%
