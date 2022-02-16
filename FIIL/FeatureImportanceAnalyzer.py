from ._get_features_importance import get_features_importance
from ._get_data import _get_X
from ._get_data import _get_y
from ._get_features_names import _get_features_names_from_data

class FeatureImportanceAnalyzer:

	def __init__(self, model, file, **params):
		#_check_X_type_and_shape(X_test)
		#_check_Y_type_and_shape(Y_test)
		
		self.model = model
		self.X = _get_X(model, file)
		self.Y = _get_y(model, file)
		self.features = _get_features_names_from_data(file)
		#self.metric = _get_metric(**params)

		self.n_simulations = params.get("n_simulations", 100)
		#self.pred_fn = params.get("pred_fn", None)
		#self.pixel_percentage = params.get("pixel_percentage", 0.02)

	def get(self):
		self.features_importance = get_features_importance(**self.__dict__)
		return self.features_importance

	#def get_metric_value(self, model, file, **params):


	#def draw(self):
		##TODO: develop some beautiful figures based on the feature importances

