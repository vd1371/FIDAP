

def _get_metric_fn(model, modelling_type, **params):

	metric = params.get("metric_fn")

	if metric is None:

		from sklearn.metrics import accuracy_score
		from sklearn.metrics import r2_score
		from sklearn.metrics import silhouette_score

		default_metrics = {'classification': accuracy_score,
							'regression': r2_score,
							'clustering': silhouette_score}
		return default_metrics[modelling_type]

	elif callable(metric):
		return metric

	elif isinstance(metric, str):
		import sklearn
		"https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter"
		return sklearn.metrics.get_scorer(metric)._score_func

	else:
		raise TypeError("The metric parameters should be None, callable, or str")