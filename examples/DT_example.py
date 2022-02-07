import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from FIIL import FeatureImportanceAnalyzer

def DT_example():

	# Loading Pima Indians Diabetes Dataset
	path = "https://raw.githubusercontent.com/npradaschnor/" \
			"Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
	pima = pd.read_csv(path)
	pima.head()

	X = pima.iloc[: , :-1].values
	y = pima.iloc[: , -1].values

	# Creating train and test datasets (70% train, 30% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30)

	# Define model
	classifier = DecisionTreeClassifier()
	classifier.fit(X_train,y_train)

	# Predict
	y_pred = classifier.predict(X_test)

	# Outputs
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