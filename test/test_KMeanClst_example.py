import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from FIDAP import FeatureImportanceAnalyzer

class KMeanClst_example(unittest.TestCase):

	def test_KMeanClst_example(self):

		# Loading data
		rand = int(np.random.uniform(1,1000))
		data, true_labels = make_blobs(n_samples=[300, 350, 250], n_features=4, random_state=rand)

		file = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4"])
		X = file.iloc[: , :].values

		# Finding optimum number of clusters
		range_n_clusters = [2, 3, 4, 5, 6]
		silhouette_avg_dict = {}

		for n_clusters in range_n_clusters:

			clusterer = KMeans(n_clusters=n_clusters, random_state=105)
			cluster_labels = clusterer.fit_predict(X)

			# The silhouette_score gives the average value for all the samples.
			# This gives a perspective into the density and separation of the formed
			# clusters
			silhouette_avg = silhouette_score(X, cluster_labels)
			#print("For n_clusters =", n_clusters,
			#"The average silhouette_score is :", silhouette_avg)
			silhouette_avg_dict[n_clusters]= silhouette_avg

		best_n_clusters = max(silhouette_avg_dict, key=silhouette_avg_dict.get)
		#print("The optimum number of clusters is equal to : ",best_n_clusters)

		# Defining model
		model = KMeans(n_clusters=best_n_clusters, random_state=105, n_init="auto")
		cluster_labels = model.fit_predict(X)

		# Outputs
		clustring_silhouette_score = silhouette_score(X, cluster_labels)
		print("Model silhouette score : ", clustring_silhouette_score)

		FIDAP = FeatureImportanceAnalyzer(model, X)
		FIDAP.get(verbose=True)
		FIDAP.boxplot()