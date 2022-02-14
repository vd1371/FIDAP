import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from FIIL import FeatureImportanceAnalyzer

def MLP_example():

	# Loading Pima Indians Diabetes Dataset
	path = "https://raw.githubusercontent.com/npradaschnor/"\
			"Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
	
	file = pd.read_csv(path)
	file.head()

	X = file.iloc[: , :-1].values
	y = file.iloc[: , -1].values

	# Creating train and test datasets (70% train, 30% test)
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

	# Define model
	classifier = MLPClassifier(hidden_layer_sizes=(8,8,8), 
	                           activation='relu', solver='adam', max_iter=1000)
	classifier.fit(X_train,y_train)

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