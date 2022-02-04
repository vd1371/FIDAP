import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from FIIL import FeatureImportanceAnalyzer

def KNN_example():
	# Loading iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target

	# Creating train and test datasets (70% train, 30% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.30)

	# Define model
	classifier = KNeighborsClassifier(n_neighbors = 10)
	classifier.fit(X_train,y_train)

	# Predict
	y_pred = classifier.predict(X_test)

	# Outputs
	print(type(classifier))
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