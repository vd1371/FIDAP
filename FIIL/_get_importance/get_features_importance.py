import numpy as np
import itertools

def get_features_importance(**params):

	model = params.get("model")
	X = params.get("X")
	Y_true = params.get("Y")
	metric_fn = params.get("metric_fn")
	n_simulations = params.get("n_simulations")
	features = params.get("features")
	pred_fn = params.get("pred_fn")
	n_feature_combination = params.get("n_feature_combination")

	y_pred = getattr(model, pred_fn)(X)
	initial_metric = metric_fn(Y_true, y_pred)

	feature_importances = {}
	feature_importances_instaces = {}

	indices = [i for i in range(len(features))]
	for n in range(1, n_feature_combination+1):

		for comb in itertools.combinations(indices, n):
				
			X_temp = X.copy()

			temp_metric_list = []
			for j in range (n_simulations):

				np.random.shuffle(X_temp[:, comb])

				y_pred = getattr(model, pred_fn)(X_temp)
				loss = initial_metric - metric_fn(Y_true, y_pred)
				temp_metric_list.append(loss)

			ft_importance = round(np.mean(temp_metric_list), 4)

			features_names = "-".join([features[i] for i in comb])

			feature_importances[features_names] = ft_importance
			feature_importances_instaces[features_names] = temp_metric_list[:]

	return feature_importances, feature_importances_instaces