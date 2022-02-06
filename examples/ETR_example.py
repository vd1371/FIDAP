import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from FIIL import FeatureImportanceAnalyzer

def ETR_example():

	# Loading boston dataset
	X, y = load_boston(return_X_y=True)

	# Creating train and test datasets (70% train, 30% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

	# Define model
	classifier = ExtraTreesRegressor()

	# Fit the model
	classifier.fit(X_train, y_train)

	# Outputs
	y_pred = classifier.predict(X_test)
	error = mean_squared_error(y_test, y_pred)
	print(error)

	# calculating feature importance
	n_features = X_train.shape[1]
	n_simulations = 10

	fiil = FeatureImportanceAnalyzer(classifier, X_test, y_test)
	print (fiil.get())

	feature_importances_ = []