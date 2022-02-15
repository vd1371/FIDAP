import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from FIIL import FeatureImportanceAnalyzer

def PA_example():

    # Loading iris Dataset
    #iris = load_iris()
    #X, y = iris.data, iris.target

    path = "https://gist.githubusercontent.com/netj/8836201/raw/" \
            "6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    file = pd.read_csv(path)

    X = file.iloc[: , :-1].values
    y = file.iloc[: , -1].values

    # Creating train and test datasets (70% train, 30% test)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

    # Define model
    Model = PassiveAggressiveClassifier(C = 0.5, random_state = 5)

    # Fit the model
    Model.fit(X_train,y_train)

    # Predict
    y_pred = Model.predict(X_test)

    # Outputs
    report = classification_report(y_test, y_pred)
    print ("Classification report", report)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Model accuracy score :", accuracy)

    # calculating feature importance
    n_features = X_train.shape[1]
    n_simulations = 10

    fiil = FeatureImportanceAnalyzer(Model, file)
    print (fiil.get())