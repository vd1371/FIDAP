from sklearn.linear_model import LinearRegression
from ._load_data_for_regression import _load_data_for_regression
from ._analyze import _analyze

def LR_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_regression()

	# Define model
	Model = LinearRegression()

	# Fit the model
	Model.fit(X_train, y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)