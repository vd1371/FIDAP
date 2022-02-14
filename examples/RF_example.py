import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from FIIL import FeatureImportanceAnalyzer

def RF_example():

    # Loading iris dataset
    #iris = datasets.load_iris()
    path = "https://gist.githubusercontent.com/netj/8836201/raw/" \
            "6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    file = pd.read_csv(path)
    
    # Pre process
    X = X = file.iloc[: , :-1].values
    y = file.iloc[: , -1].values
    #X = iris.data[:, :]
    #y = iris.target

    # Creating train and test datasets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.30)

    # Define model
    classifier = RandomForestClassifier(n_estimators = 50)
    classifier.fit(X_train, y_train)

    # Predict
    y_pred = classifier.predict(X_test)

    # Outputs
    report = classification_report(y_test, y_pred)
    print ("Classification report", report)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Model accuracy score :", accuracy)

    # calculating feature importance
    n_features = X_train.shape[1]
    n_simulations = 10

    fiil = FeatureImportanceAnalyzer(classifier, file)
    print (fiil.get())