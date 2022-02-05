import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from FIIL import FeatureImportanceAnalyzer

def NB_example():

	# Loading Pima Indians Diabetes Dataset
	dataset = load_boston()
	X = dataset.data
	y = dataset.target

	# Creating train and test datasets (70% train, 30% test)
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

	# Define model
	classifier = BayesianRidge()
	classifier.fit(X_train,y_train)

	# Predict
	y_pred = classifier.predict(X_test)

	# Outputs
	r2 = r2_score(y_test, y_pred)
	print(f"r2 Score Of Test Set : {r2}")

	# calculating feature importance
	n_features = X_train.shape[1]
	n_simulations = 10

	fiil = FeatureImportanceAnalyzer(classifier, X_test, y_test)
	print (fiil.get())

	feature_importances_ = []