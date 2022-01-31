import sklearn

from ._get_features_imporatnce_for_RF_CL import _get_features_imporatnce_for_RF_CL

def get_features_importance(**params):

	model = params.get("model")

	if isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
		return _get_features_imporatnce_for_RF_CL(**params)