from ._get_features_importance import get_features_importance

class FeatureImportanceAnalyzer:

	def __init__(self, model, X_test, Y_test, **params):
		self.model = model
		self.X = X_test
		self.Y = Y_test
		pass

	def get(self):
		self.features_importance = get_features_importance(**self.__dict__)

