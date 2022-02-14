import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from FIIL import FeatureImportanceAnalyzer

def DTR_example():

	# Loading Boston Dataset
	path = "https://raw.githubusercontent.com/selva86/datasets/" \
            "master/BostonHousing.csv"
	file = pd.read_csv(path)

	X = file.iloc[: , :-1].values
	y = file.iloc[: , -1].values

	# Creating train and test datasets (70% train, 30% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

	# Define model
	classifier = DecisionTreeRegressor(random_state = 0)

	# Fit the model
	classifier.fit(X_train, y_train)

	# Outputs
	y_pred = classifier.predict(X_test)
	r2 = r2_score(y_test, y_pred)
	print(f"Model R2 Score : {r2}")


	# calculating feature importance
	n_features = X_train.shape[1]
	n_simulations = 10

	fiil = FeatureImportanceAnalyzer(classifier, file)
	print (fiil.get())