from ._get_importance import get_features_importance
from ._check_X_Y_type_and_shape import _check_X_Y_type_and_shape
from ._prepare_X_Y_features import _prepare_X_Y_features
from ._get_metric_fn import _get_metric_fn
from ._plot_box_and_save import _plot_box_and_save
from ._summarize import _summarize
from ._get_string_report import _get_string_report


class FeatureImportanceAnalyzer:

	def __init__(self, model, X, Y = None, **params):
		'''
		Feature Importance by DAta Permutation

		The idea is based on the data permutation proposed by Dr. Breiman in
		his famous Random Forest paper.

		Parameters
		----------
		model: callable
			The prediction or clustering model


		X: pandas.Dataframe, list of lists, numpy 2D array
			The input variables as pandas dataframe

		Y: pandas.Dataframe, pandas.Seris, list, numpy array
			The output variable. It must be one dimensional. If one-hot encoding
			is used for the multiclass classification as the output variable,
			only one of the outputs must be passed.

		features: list, default = None
			length of passed features must be equal to number of input variables

			if None:
				the X.columns will be used as features if X is a pd.DataFrame
				"X{i}" will be used, e.g., X0, X1, X2...

		metric_fn: str or callable, default = None
			
			The metric_fn needs to be ascending

			if None:
				the default metric function will be used for the metric_fn
					accuracy for classification
					R2 for regression
					silhouette for clustering

			if callable:
				the passed function will be used. The 
			
			if str:
				sklearn.metrics.get_scorer will be called
				more info: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
		
		n_simulations: int, default = 10
			the number of random permutation for each feature
			min (n_simulations) = 10

		pred_fn: str, default = "predict"
			The prediction method of the passed model. The default value is 
			"predict"

		direct: str, default = "."
			the directory for saving the summary and the figures. Default:
			root directory

		verbose: {True, False}, default = False
			verbosity

		n_feature_combination: int, default = 1
			number of features to be permutated for analysis
			Default is 1 and max is len(features)

		output_fig_format: str, default = 'jpg'
			the format of the figure to be saved, default is ".tif"


		Attributes
		----------
		features_importance: {features: average of feature importance vals}
			the average of the feature importance for features

		features_importance_instances: {feature: [instances of feature importances]}
			the instances of features importance for features

		Methods:
		----------
		get():
			returns feature_importance

		boxplot():
			saves the boxplot figure into the directory passed by user

		summary():
			saves the report into the directory passed by user

		run():
			boxplot() + summary()

		__str__():
			return the string format of features importances

		References
	    ----------
	    ... L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

		Examples
		----------
		>>> from FIDAP import FeatureImportanceAnalyzer
		>>> fidap = FeatureImportanceAnalyzer(Model, X_test, y_test,
        ...                                   n_feature_combination = 3,
        ...                                   n_simulations = 10)
		>>> fidap.run()
		>>> print (fidap)
		Feature                              FIDAP  
		--------------------------------------------
		sepal length (cm)                    0.0000
		sepal width (cm)                     0.0000
		petal length (cm)                    0.2022
		petal width (cm)                     0.1689
		--------------------------------------------
		'''
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
		self.output_fig_format = params.get("output_fig_format", 'jpg')

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

	def __str__(self):
		return _get_string_report(self.features_importance)

	def run(self):
		self.boxplot()
		self.summary()

