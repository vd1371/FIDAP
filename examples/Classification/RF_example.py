from sklearn.ensemble import RandomForestClassifier
from ._load_data_for_classification import _load_data_for_classification
from ._analyze import _analyze

from FIIL import FeatureImportanceAnalyzer
import pprint

def RF_example():

    # Loading dataset
    X_train, X_test, y_train, y_test = _load_data_for_classification()

    # Define model
    Model = RandomForestClassifier(n_estimators = 50)

    # Fit the model
    Model.fit(X_train, y_train)

    # Feature importance analysis
    fiil = FeatureImportanceAnalyzer(Model,
                                    X_test,
                                    y_test,
                                    n_feature_combination = 2,
                                    n_simulations = 10)
    pprint.pprint(fiil.get())
    fiil.boxplot()
    fiil.summary()