try: 
	import keras
except:
	from tensorflow import keras

import sklearn
import xgboost
import catboost

def _get_type_of_modelling(model)
	if isinstance(model, 
				(sklearn.linear_model._bayes.BayesianRidge,
				sklearn.tree._classes.DecisionTreeRegressor,
				keras.engine.sequential.Sequential,
				sklearn.ensemble._forest.ExtraTreesRegressor,
				sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor,
				xgboost.sklearn.XGBRegressor,
				sklearn.svm._classes.SVR,
				sklearn.ensemble._gb.GradientBoostingRegressor,
				sklearn.linear_model._base.LinearRegression)):
		return 'regression'

	elif isinstance(model, 
				(sklearn.ensemble._forest.RandomForestClassifier,
				sklearn.neighbors._classification.KNeighborsClassifier,
				sklearn.linear_model._logistic.LogisticRegression,
				sklearn.svm._classes.SVC,
				sklearn.neural_network._multilayer_perceptron.MLPClassifier,
				sklearn.tree._classes.DecisionTreeClassifier,
				sklearn.ensemble._forest.ExtraTreesClassifier,
				sklearn.neighbors._classification.RadiusNeighborsClassifier,
				sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier,
				sklearn.ensemble._gb.GradientBoostingClassifier,
				catboost.core.CatBoostClassifier)):
		return 'classification'

	elif isinstance (model,
						(sklearn.cluster._kmeans.KMeans,
						sklearn.cluster._mean_shift.MeanShift)):
		return 'clustering'