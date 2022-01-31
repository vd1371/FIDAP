from ._get_features_importance import get_features_importance

class FeatureImportanceAnalyzer:

	def __init__(self, model, X_test, Y_test, **params):
		self.model = model
		self.X = X_test
		self.Y = Y_test

		self.n_simulations = params.get("'n_simulations", 100)

	def get(self):
		self.features_importance = get_features_importance(**self.__dict__)
		return self.features_importance

