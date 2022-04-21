

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer

from catboost import CatBoostClassifier

def CBC_example():

	# Loading dataset
	data = load_iris()

	df = pd.DataFrame(data=data.data, columns=data.feature_names)
	df['Y'] = data.target

	X = df.iloc[: , :-1]
	y = df.iloc[: , -1]

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y,test_size=0.30)

	# Define model
	Model = CatBoostClassifier()

	# Fit the model
	Model.fit(X_train, y_train)

	# Feature importance analysis
	FIDAP = FeatureImportanceAnalyzer(*args, **kwargs)
	pprint.pprint(FIDAP.get())
	FIDAP.boxplot()