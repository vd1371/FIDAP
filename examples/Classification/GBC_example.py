from sklearn.ensemble import GradientBoostingClassifier
from ._load_data_for_classification import _load_data_for_classification
from ._analyze import _analyze

def GBC_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_classification()

	# Define model
	Model = GradientBoostingClassifier()

	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)