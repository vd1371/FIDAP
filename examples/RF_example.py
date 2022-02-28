from sklearn.ensemble import RandomForestClassifier
from ._load_data_for_classification import _load_data_for_classification
from FIIL import FeatureImportanceAnalyzer
from ._analyze import _analyze

def RF_example():

    # Loading dataset
    X_train, X_test, y_train, y_test = _load_data_for_classification()

    # Define model
    Model = RandomForestClassifier(n_estimators = 50)

    # Fit the model
    Model.fit(X_train, y_train)

    _analyze(Model, X_test, y_test, features = None)