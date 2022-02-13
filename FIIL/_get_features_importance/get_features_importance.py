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

	X = params.get("X")
	Y_true = params.get("Y")
	n_simulations = params.get("n_simulations")
	features = features = [f"X{i}" for i in range(X.shape[1])] #params.get("features")
	#file = params.get("file")
	#metric = params.get("metric")
	model = params.get("model")
	pred_fn = params.get("pred_fn")

	y_pred = model.predict(X)
	initial_metric = _metric_value(model, X, Y_true)._get()

	feature_importances_ = {}
	for i, feature in enumerate(features):
		
		X_temp = X.copy()

		temp_metric_list = []

		for j in range (n_simulations):

			np.random.shuffle(X_temp[:,i])

			y_pred = model.predict(X_temp)

			metric_temp = initial_metric - _metric_value(model, X_temp, Y_true)._get()

			temp_metric_list.append(metric_temp)

		ft_importance = np.mean(temp_metric_list)
		# ft_importance = abs(metric_mean) if metric_mean < 0 else 0
		feature_importances_[feature] = ft_importance

	return feature_importances_