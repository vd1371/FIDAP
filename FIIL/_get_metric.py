import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

from ._get_type_of_modelling import _get_type_of_modelling

def _get_metric(**params):

	metric = params.get("metric")

	if metric == None:
		modelling_type = _get_type_of_modelling(model)

		default_metrics = {'classification': accuracy_score,
							'regression': r2_score,
							'clustering': silhouette_score}
		return default_metrics[modelling_type]

	elif callable(metric):
		return metric

	elif isinstance(metric, str):
		"https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter"
		return sklearn.metrics.get_scorer(metric)