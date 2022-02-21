import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from ._load_data_for_regression import _load_data_for_regression
from FIIL import FeatureImportanceAnalyzer

def XGB_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_regression()

	# Define model
	Model = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3,
                              learning_rate = 0.1, max_depth = 20, alpha = 10, 
                              n_estimators = 40)
	
	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	fiil = FeatureImportanceAnalyzer(Model,
									X_test,
									y_test,
									features = None)
	print (fiil.get())