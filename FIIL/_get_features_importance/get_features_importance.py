import sklearn
import keras

from ._get_features_imporatnce_for_RF_CL import _get_features_imporatnce_for_RF_CL
from ._get_features_imporatnce_for_DNN_RE import _get_features_imporatnce_for_DNN_RE

def get_features_importance(**params):

	model = params.get("model")

	if isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
		return _get_features_imporatnce_for_RF_CL(**params)
	elif isinstance(model, keras.engine.sequential.Sequential):
		return _get_features_imporatnce_for_DNN_RE(**params)