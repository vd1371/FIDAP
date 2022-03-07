from sklearn.neighbors import RadiusNeighborsClassifier
from ._load_data_for_classification import _load_data_for_classification
from ._analyze import _analyze

def RN_example():

    # Loading dataset
    X_train, X_test, y_train, y_test = _load_data_for_classification()

    # Define model
    Model = RadiusNeighborsClassifier(algorithm='auto', leaf_size=30, 
                                            metric='minkowski',
                                            metric_params=None, n_jobs=None, 
                                            outlier_label=None, p=2, 
                                            radius=3.5, weights='uniform')

    # Fit the model
    Model.fit(X_train,y_train)

    # Feature importance analysis
    _analyze(Model, X_test, y_test, features = None)