import numpy as np


def get_features_imporatnce(**params):

	X = params.get("X")
	Y_true = params.get("Y")
	n_simulations = params.get("n_simulations")
	features = params.get("features")
	metric = params.get("metric")
	model = params.get("model")
	pred_fn = params.get("pred_fn")

	y_pred = _get_preds(X, model, pred_fn)
	initial_metric = metric(y_pred, Y_true)

	feature_importances_ = {}
	for i, feature in enumerate(features):
		
		X_temp = X.copy()

		temp_metric_list = []

		for j in range (n_simulations):

			np.random.shuffle(X_temp[:,i])

			y_pred = _get_preds(X_temp, model, pred_fn)
			metric_temp = initial_metric - metric(y_pred, Y_true)

			temp_metric_list.append(metric_temp)

		ft_importance = np.mean(temp_metric_list)
		# ft_importance = abs(metric_mean) if metric_mean < 0 else 0
		feature_importances_[feature] = ft_importance

	return feature_importances_