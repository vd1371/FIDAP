import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from FIDAP import FeatureImportanceAnalyzer

class MSClst_example(unittest.TestCase):

	def test_MSClst_example(self):

		rand = int(np.random.uniform(1,1000))

		data, true_labels = make_blobs(n_samples=[300,350,250],
									   n_features=4, random_state=rand)

		file = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4"])
		#file.plot.scatter("x1", "x2")

		X = file.iloc[: , :].values

		bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

		model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
		model.fit(X)
		cluster_labels = model.labels_
		clustring_silhouette_score = silhouette_score(X, cluster_labels)

		# calculating feature importance
		FIDAP = FeatureImportanceAnalyzer(model, X)
		FIDAP.get(verbose=True)
		FIDAP.boxplot()