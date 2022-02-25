import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from ._load_data_for_regression import _load_data_for_regression
from FIIL import FeatureImportanceAnalyzer

def DTR_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_regression()

	# Define model
	Model = DecisionTreeRegressor(random_state = 0)

	# Fit the model
	Model.fit(X_train, y_train)

	# Feature importance analysis
	fiil = FeatureImportanceAnalyzer(Model,
									X_test,
									y_test,
									features = None)
	print (fiil.get())