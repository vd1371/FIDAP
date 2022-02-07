try: 
	import keras
except:
	from tensorflow import keras

import sklearn
from sklearn.metrics import accuracy_score

def _get_metric(**params):

	metric_fn = params.get("metric_fn")
	metric = params.get("metric")

	if not metric_fn == None:
		return metric_fn

	elif metric in ['acc', "accuracy"]:
		return accuracy_score


	else:

		if isinstance(model, 
						(sklearn.ensemble._forest.RandomForestClassifier,
						sklearn.neighbors._classification.KNeighborsClassifier,
						sklearn.linear_model._logistic.LogisticRegression,
						sklearn.svm._classes.SVC,

		return accuracy_score

			)):
		elif isinstance(model, ):
			return _get_features_imporatnce_for_KNN_CL(**params)
		elif isinstance(model, ):
			return _get_features_imporatnce_for_LogR_CL(**params)
		elif isinstance(model, ):
			return _get_features_imporatnce_for_SVM_CL(**params)
		elif isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPClassifier):
			return _get_features_imporatnce_for_MLP_CL(**params)
		elif isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
			return _get_features_imporatnce_for_DT_CL(**params)
		elif isinstance(model, sklearn.linear_model._bayes.BayesianRidge):
			return _get_features_imporatnce_for_NB_CL(**params)
		elif isinstance(model, sklearn.ensemble._forest.ExtraTreesClassifier):
			return _get_features_imporatnce_for_ET_CL(**params)
		
		
		elif isinstance(model, sklearn.tree._classes.DecisionTreeRegressor):
			return _get_features_imporatnce_for_DT_RE(**params)
		elif isinstance(model, sklearn.ensemble._forest.ExtraTreesRegressor):
			return _get_features_imporatnce_for_ET_RE(**params)
		elif isinstance(model, keras.engine.sequential.Sequential):
			return _get_features_imporatnce_for_DNN_RE(**params)

	# return accuracy for classification problems
	# return R2 for regression problems
	# return a standard metric for clustering

