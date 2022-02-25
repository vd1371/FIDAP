from ._get_features_importance import get_features_importance
from ._check_X_Y_type_and_shape import _check_X_Y_type_and_shape
from ._prepare_X_Y_features import _prepare_X_Y_features
from ._get_metric_fn import _get_metric_fn
from ._plot_box_and_save import _plot_box_and_save


class FeatureImportanceAnalyzer:

	def __init__(self, model, X, Y = None, **params):
		_check_X_Y_type_and_shape(X, Y)
		
		self.model = model
		self.X, self.Y, self.features = \
			_prepare_X_Y_features(model, X, Y, **params)
		self.metric_fn = _get_metric_fn(model, **params)
		self.n_simulations = params.get("n_simulations", 100)
		self.pred_fn = params.get("pred_fn", "predict")
		self.direc = params.get("direc", '.')

	def get(self):
		self.features_importance, \
		self.features_importance_instances, \
		self.features = \
			get_features_importance(**self.__dict__)
		_plot_box_and_save(get_features_importance(**self.__dict__)[1], 
							get_features_importance(**self.__dict__)[2])

		return self.features_importance

	#def draw(self):
		##TODO: develop some beautiful figures based on the feature importances

