import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from FIIL import FeatureImportanceAnalyzer

def XGB_example():

	# Loading Boston Dataset
	path = "https://raw.githubusercontent.com/selva86/datasets/" \
            "master/BostonHousing.csv"
	file = pd.read_csv(path)

	X = file.iloc[: , :-1].values
	y = file.iloc[: , -1].values

	#dataset = load_boston()
	#X = dataset.data
	#y = dataset.target

	# Creating train and test datasets (70% train, 30% test)
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

	# Define model
	Model = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3,
                              learning_rate = 0.1, max_depth = 20, alpha = 10, 
                              n_estimators = 40)
	Model.fit(X_train,y_train)

	# Predict
	y_pred = Model.predict(X_test)

	# Outputs
	r2 = r2_score(y_test, y_pred)
	print(f"Model R2 Score : {r2}")

	# calculating feature importance
	n_features = X_train.shape[1]
	n_simulations = 10

	fiil = FeatureImportanceAnalyzer(Model, file)
	print (fiil.get())