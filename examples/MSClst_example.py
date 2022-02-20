import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

# Loading data
rand = int(np.random.uniform(1,1000))
    
data, true_labels = make_blobs(n_samples=[300,350,250], 
                               n_features=4, random_state=rand)

file = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4"])
#file.plot.scatter("x1", "x2")
#print(file)
X = file.iloc[: , :].values

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

Model = MeanShift(bandwidth=bandwidth, bin_seeding=True)

Model.fit(X)
labels = Model.labels_
print(type(Model))
clustring_silhouette_score = silhouette_score(X, labels)
print(clustring_silhouette_score)

cluster_centers = Model.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("The optimum number of clusters is equal to : %d" % n_clusters_)