from ._get_importance import get_pixel_importance
from ._get_metric_fn import _get_metric_fn

class PixelImportanceAnalyzer:

	def __init__(self, model, **params):
		
		self.model = model
		self.metric_fn = "predict"
		self.n_simulations = params.get("n_simulations", 100)
		self.pred_fn = params.get("pred_fn", "predict")
		self.direc = params.get("direc", '.')

	def get(self, img, label):
		self.pixel_importance = get_pixel_importance(img, **self.__dict__)
		return self.pixel_importance

	def boxplot(self):
		if not hasattr(self, 'features_importance_instances'):
			self.get()
		_plot_box_and_save(self.features_importance_instances)

