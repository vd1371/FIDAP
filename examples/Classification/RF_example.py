from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer
import pprint

def RF_example():

    # Loading dataset
    data = load_iris()

    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['Y'] = data.target

    X = df.iloc[: , :-1]
    y = df.iloc[: , -1]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,test_size=0.30)

    # Define model
    Model = RandomForestClassifier(n_estimators = 50)

    # Fit the model
    Model.fit(X_train, y_train)

    # Feature importance analysis
    fidap = FeatureImportanceAnalyzer(Model, X_test, y_test,
                                    n_feature_combination = 2,
                                    n_simulations = 20)
    
    fidap.run()

    print (fidap)