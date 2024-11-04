import numpy as np
import itertools
import time

def get_features_importance(**params):

	model = params.get("model")
	X = params.get("X")
	Y_true = params.get("Y")
	metric_fn = params.get("metric_fn")
	n_simulations = params.get("n_simulations")
	features = params.get("features")
	pred_fn = params.get("pred_fn")
	n_feature_combination = params.get("n_feature_combination")
	verbose = params.get("verbose")
	modelling_type = params.get("modelling_type")

	if modelling_type in ["classification", "regression"]:
		y_pred = getattr(model, pred_fn)(X)
		initial_metric = metric_fn(Y_true, y_pred)
	else:
		cluster_labels = model.labels_
		initial_metric = metric_fn(X, cluster_labels)

	feature_importances = {}
	feature_importances_instaces = {}

	indices = [i for i in range(len(features))]
	for n in range(1, n_feature_combination+1):

		n_combinations = len(list(itertools.combinations(indices, n)))
		for iter, comb in enumerate(itertools.combinations(indices, n)):

			if verbose:
				print (f"Feature {comb} are about to be analyzed")

			X_temp = X.copy()

			temp_metric_list = []
			for j in range (n_simulations):

				print (f"{j+1}/{n_simulations} simulation | {iter+1}/{n_combinations} combinations " + \
					   f"| {n+1}/{n_feature_combination} features", end="\r")

				for col in comb:
					np.random.shuffle(X_temp[:, col])

				if modelling_type in ["classification", "regression"]:
					y_pred = getattr(model, pred_fn)(X_temp)
					loss = initial_metric - metric_fn(Y_true, y_pred)

				else:
					model.fit(X_temp)
					cluster_labels = model.labels_
					loss = initial_metric - metric_fn(X, cluster_labels)

				temp_metric_list.append(loss)

			ft_importance = round(np.mean(temp_metric_list), 4)

			features_names = f"F{str(comb)}-" + "-".join([features[i] for i in comb])

			feature_importances[features_names] = ft_importance
			feature_importances_instaces[features_names] = temp_metric_list[:]

	return feature_importances, feature_importances_instaces