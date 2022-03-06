from sklearn.neighbors import KNeighborsClassifier
from ._load_data_for_classification import _load_data_for_classification
from ._analyze import _analyze

def KNN_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_classification()

	# Define model
	Model = KNeighborsClassifier(n_neighbors = 10)

	# Fit the model
	Model.fit(X_train, y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)