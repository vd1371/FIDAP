try: 
	import keras
except:
	from tensorflow import keras

import numpy as np
from FIIL._get_metric import _metric_value
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

def get_features_importance(**params):

	model = params.get("model")
	X = params.get("X")
	Y_true = params.get("Y")
	metric = params.get("metric")
	n_simulations = params.get("n_simulations")
	features = params.get("features")
	pred_fn = params.get("pred_fn")

	y_pred = getattr(model, pred_fn)(X)
	initial_metric = metric(Y_true, y_pred)

	feature_importances = {}
	for i, feature in enumerate(features):
			
		X_temp = X.copy()

		temp_metric_list = []
		for j in range (n_simulations):
			np.random.shuffle(X_temp[:,i])
			y_pred = getattr(model, pred_fn)(X_temp)
			loss = initial_metric - metric(Y_true, y_pred)
			temp_metric_list.append(loss)

		ft_importance = np.mean(temp_metric_list)
		feature_importances[feature] = ft_importance

	return feature_importances