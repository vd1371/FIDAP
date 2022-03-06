from sklearn.svm import SVR
from ._load_data_for_regression import _load_data_for_regression
from ._analyze import _analyze

def SVR_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_regression()

	# Define model
	Model = SVR( kernel='linear')

	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)