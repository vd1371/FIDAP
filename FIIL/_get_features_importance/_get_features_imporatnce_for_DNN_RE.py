import numpy as np

def _get_features_imporatnce_for_DNN_RE(**params):

	X = params.get("X")
	Y_true = params.get("Y")
	model = params.get("model")
	n_simulations = params.get("n_simulations")

	# features = _get_features_from_data(x)
	features = [f"X{i}" for i in range(X.shape[1])]

	initial_metric = model.evaluate(X_test, y_test, verbose=0)

	feature_importances_ = {}
	for i, feature in enumerate(features):
		X_temp = X.copy()

		temp_metric_list = []

		for j in range (n_simulations):

			np.random.shuffle(X_temp[:,i])

			new_metric = model.evaluate(X_temp, y_test, verbose=0)

			metric_temp = initial_metric - new_metric

			temp_metric_list.append(metric_temp)

		metric_mean = round(np.mean(temp_metric_list), 4)
		ft_importance = abs(metric_mean) if metric_mean < 0 else 0
		feature_importances_[feature] = ft_importance

	return feature_importances_