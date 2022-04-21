from ._get_importance import get_features_importance
from ._check_X_Y_type_and_shape import _check_X_Y_type_and_shape
from ._prepare_X_Y_features import _prepare_X_Y_features
from ._get_metric_fn import _get_metric_fn
from ._plot_box_and_save import _plot_box_and_save
from ._summarize import _summarize


class FeatureImportanceAnalyzer:

	def __init__(self, model, X, Y = None, **params):
		_check_X_Y_type_and_shape(X, Y)
		
		self.model = model
		self.X, self.Y, self.features = \
			_prepare_X_Y_features(model, X, Y, **params)
		self.metric_fn = _get_metric_fn(model, **params)
		self.n_simulations = max(params.get("n_simulations", 100), 10)
		self.pred_fn = params.get("pred_fn", "predict")
		self.direc = params.get("direc", '.')
		self.verbose = params.get("verbose", False)

		self.n_feature_combination = min(params.get("n_feature_combination", 1),
										len(self.features))
		self.output_fig_format = params.get("output_fig_format", 'tif')

	def get(self):
		self.features_importance, self.features_importance_instances = \
				get_features_importance(**self.__dict__)

		return self.features_importance

	def boxplot(self):
		if not hasattr(self, 'features_importance_instances'):
			self.get()
		_plot_box_and_save(**self.__dict__)

	def summary(self):
		if not hasattr(self, 'features_importance_instances'):
			self.get()
		_summarize(**self.__dict__)

