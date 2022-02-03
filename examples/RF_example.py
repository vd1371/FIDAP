import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from FIIL import FeatureImportanceAnalyzer


def RF_example():
    # iris dataset
    iris = datasets.load_iris()

    # pre process
    X = iris.data[:, :]
    y = iris.target

    # creating train and test datasets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.30)
    classifier = RandomForestClassifier(n_estimators = 50)
    classifier.fit(X_train, y_train)

    # predict
    y_pred = classifier.predict(X_test)

    # outputs
    report = classification_report(y_test, y_pred)
    print ("Classification report", report)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy score", accuracy)

    # calculating feature importance

    n_features = X_train.shape[1]
    n_simulations = 10

    fiil = FeatureImportanceAnalyzer(classifier, X_test, y_test)
    print (fiil.get())

    feature_importances_ = []