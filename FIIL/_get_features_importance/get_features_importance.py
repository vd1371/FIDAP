import sklearn
import keras

from ._get_features_imporatnce_for_RF_CL import _get_features_imporatnce_for_RF_CL
from ._get_features_imporatnce_for_DNN_RE import _get_features_imporatnce_for_DNN_RE
from ._get_features_imporatnce_for_KNN_CL import _get_features_imporatnce_for_KNN_CL
from ._get_features_imporatnce_for_LogR_CL import _get_features_imporatnce_for_LogR_CL
from ._get_features_imporatnce_for_SVM_CL import _get_features_imporatnce_for_SVM_CL
from ._get_features_imporatnce_for_MLP_CL import _get_features_imporatnce_for_MLP_CL
from ._get_features_imporatnce_for_DT_CL import _get_features_imporatnce_for_DT_CL
from ._get_features_imporatnce_for_NB_CL import _get_features_imporatnce_for_NB_CL
from ._get_features_imporatnce_for_DT_RE import _get_features_imporatnce_for_DT_RE
from ._get_features_imporatnce_for_ET_CL import _get_features_imporatnce_for_ET_CL
from ._get_features_imporatnce_for_ET_RE import _get_features_imporatnce_for_ET_RE

def get_features_importance(**params):

	model = params.get("model")

	if isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
		return _get_features_imporatnce_for_RF_CL(**params)
	elif isinstance(model, keras.engine.sequential.Sequential):
		return _get_features_imporatnce_for_DNN_RE(**params)
	elif isinstance(model, sklearn.neighbors._classification.KNeighborsClassifier):
		return _get_features_imporatnce_for_KNN_CL(**params)
	elif isinstance(model, sklearn.linear_model._logistic.LogisticRegression):
		return _get_features_imporatnce_for_LogR_CL(**params)
	elif isinstance(model, sklearn.svm._classes.SVC):
		return _get_features_imporatnce_for_SVM_CL(**params)
	elif isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPClassifier):
		return _get_features_imporatnce_for_MLP_CL(**params)
	elif isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
		return _get_features_imporatnce_for_DT_CL(**params)
	elif isinstance(model, sklearn.linear_model._bayes.BayesianRidge):
		return _get_features_imporatnce_for_NB_CL(**params)
	elif isinstance(model, sklearn.tree._classes.DecisionTreeRegressor):
		return _get_features_imporatnce_for_DT_RE(**params)
	elif isinstance(model, sklearn.ensemble._forest.ExtraTreesClassifier):
		return _get_features_imporatnce_for_ET_CL(**params)
	elif isinstance(model, sklearn.ensemble._forest.ExtraTreesRegressor):
		return _get_features_imporatnce_for_ET_RE(**params)