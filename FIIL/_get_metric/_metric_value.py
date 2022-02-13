#try: 
	#import keras
#except:
	#from tensorflow import keras
from tensorflow import keras
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score


class _metric_value:

	def __init__(self, model, X, y_true, **params):
		self.model = model
		self.X = X
		self.y_true = y_true
		self.y_pred = model.predict(self.X)

	def _get(self):

		if isinstance(self.model, 
							(sklearn.linear_model._bayes.BayesianRidge,
							sklearn.tree._classes.DecisionTreeRegressor,
							sklearn.ensemble._forest.ExtraTreesRegressor)):
							#keras.engine.sequential.Sequential)):
			value = r2_score(self.y_true, self.y_pred) #for regression
			return value

		elif isinstance(self.model, 
							(sklearn.ensemble._forest.RandomForestClassifier,
							sklearn.neighbors._classification.KNeighborsClassifier,
							sklearn.linear_model._logistic.LogisticRegression,
							sklearn.svm._classes.SVC,
							sklearn.neural_network._multilayer_perceptron.MLPClassifier,
							sklearn.tree._classes.DecisionTreeClassifier,
							sklearn.ensemble._forest.ExtraTreesClassifier)):
			value = accuracy_score(self.y_true, self.y_pred) #for classification
			return value

