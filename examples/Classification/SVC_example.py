from sklearn.svm import SVC
from ._load_data_for_classification import _load_data_for_classification
from ._analyze import _analyze

def SVC_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_classification()

	# Define model
	Model = SVC(kernel='linear') # Linear Kernel

	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)