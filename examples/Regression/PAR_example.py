from sklearn.linear_model import PassiveAggressiveRegressor
from ._load_data_for_regression import _load_data_for_regression
from ._analyze import _analyze

def PAR_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_regression()

	# Define model
	Model = PassiveAggressiveRegressor()

	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)