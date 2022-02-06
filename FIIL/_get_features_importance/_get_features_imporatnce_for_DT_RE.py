import numpy as np
from sklearn.metrics import mean_squared_error


def _get_features_imporatnce_for_DT_RE(**params):

	X = params.get("X")
	Y_true = params.get("Y")
	model = params.get("model")
	n_simulations = params.get("n_simulations")

	features = [f"X{i}" for i in range(X.shape[1])]

	y_pred = model.predict(X)

	initial_metric = mean_squared_error(Y_true, y_pred)

	feature_importances_ = {}

	for i, feature in enumerate(features):
		X_temp = X.copy()

		temp_metric_list = []

		for j in range (n_simulations):

			np.random.shuffle(X_temp[:,i])

			y_pred = model.predict(X_temp)

			metric_temp = initial_metric - mean_squared_error(Y_true, y_pred)

			temp_metric_list.append(metric_temp)

		metric_mean = round(np.mean(temp_metric_list), 4)
		ft_importance = abs(metric_mean) if metric_mean < 0 else 0
		feature_importances_[feature] = ft_importance

	return feature_importances_