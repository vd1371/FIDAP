# dataset has to have a pandas dataframework structure
try: 
	import keras
except:
	from tensorflow import keras

import sklearn
import xgboost
import catboost
import pandas as pd
from sklearn.neural_network import MLPClassifier

def _get_y(model, file):
	if not isinstance (file, pd.core.frame.DataFrame):
		raise TypeError("The dataset must have a Pandas dataframe structure")
	elif isinstance(model, 	(sklearn.ensemble._forest.RandomForestClassifier,
							sklearn.neighbors._classification.KNeighborsClassifier,
							sklearn.linear_model._logistic.LogisticRegression,
							sklearn.svm._classes.SVC,
							sklearn.neural_network._multilayer_perceptron.MLPClassifier,
							sklearn.tree._classes.DecisionTreeClassifier,
							sklearn.ensemble._forest.ExtraTreesClassifier,
							sklearn.neighbors._classification.RadiusNeighborsClassifier,
							sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier,
							sklearn.ensemble._gb.GradientBoostingClassifier,
							catboost.core.CatBoostClassifier,
							sklearn.linear_model._bayes.BayesianRidge,
							sklearn.tree._classes.DecisionTreeRegressor,
							keras.engine.sequential.Sequential,
							sklearn.ensemble._forest.ExtraTreesRegressor,
							sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor,
							xgboost.sklearn.XGBRegressor,
							sklearn.svm._classes.SVR,
							sklearn.ensemble._gb.GradientBoostingRegressor,
							sklearn.linear_model._base.LinearRegression)):
		return file.iloc[: , -1].values