from ._get_features_importance import get_features_importance

class FeatureImportanceAnalyzer:

	def __init__(self, model, X_test, Y_test = None, **params):
		_check_X_type_and_shape(X_test)
		_check_Y_type_and_shape(Y_test)

		self.model = model
		self.X = _get_X(X_test)
		self.Y = Y_test
		self.features = _get_features_names_from_data(X_test)
		self.metric = _get_metric(**params)

		self.n_simulations = params.get("n_simulations", 100)
		self.pred_fn = params.get("pred_fn", None)
		self.pixel_percentage = params.get("pixel_percentage", 0.02)

	def get(self):
		self.features_importance = get_features_importance(**self.__dict__)
		return self.features_importance

	def draw(self):
		##TODO: develop some beautiful figures based on the feature importances

