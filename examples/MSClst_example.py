import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from ._load_data_for_clustering import _load_data_for_clustering
from sklearn.metrics import silhouette_score
from FIIL import FeatureImportanceAnalyzer

def MSClst_example():

	# Loading data
	X = _load_data_for_clustering()

	bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

	Model = MeanShift(bandwidth=bandwidth, bin_seeding=True)

	Model.fit(X)

	cluster_labels = Model.labels_

	clustring_silhouette_score = silhouette_score(X, cluster_labels)

	print("Model silhouette score : ", clustring_silhouette_score)

	#cluster_centers = Model.cluster_centers_

	# calculating feature importance

	n_features = X.shape[1]

	n_simulations = 10

	fiil = FeatureImportanceAnalyzer(Model, file)

	print (fiil.get())