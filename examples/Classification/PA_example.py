from sklearn.linear_model import PassiveAggressiveClassifier
from ._load_data_for_classification import _load_data_for_classification
from ._analyze import _analyze

def PA_example():

    # Loading dataset
    X_train, X_test, y_train, y_test = _load_data_for_classification()

    # Define model
    Model = PassiveAggressiveClassifier(C = 0.5, random_state = 5)

    # Fit the model
    Model.fit(X_train, y_train)

    # Feature importance analysis
    _analyze(Model, X_test, y_test, features = None)