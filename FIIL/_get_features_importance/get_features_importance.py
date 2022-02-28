import numpy as np
def get_features_importance(**params):

	model = params.get("model")
	X = params.get("X")
	Y_true = params.get("Y")
	metric_fn = params.get("metric_fn")
	n_simulations = params.get("n_simulations")
	features = params.get("features")
	pred_fn = params.get("pred_fn")

	y_pred = getattr(model, pred_fn)(X)
	initial_metric = metric_fn(Y_true, y_pred)

	feature_importances = {}
	feature_importances_instaces = {}
	for i, feature in enumerate(features):
			
		X_temp = X.copy()

		temp_metric_list = []
		for j in range (n_simulations):
			np.random.shuffle(X_temp[:,i])
			y_pred = getattr(model, pred_fn)(X_temp)
			loss = initial_metric - metric_fn(Y_true, y_pred)
			temp_metric_list.append(loss)

		ft_importance = round(np.mean(temp_metric_list), 4)
		feature_importances[feature] = ft_importance
		feature_importances_instaces[feature] = temp_metric_list[:]

	return feature_importances, feature_importances_instaces