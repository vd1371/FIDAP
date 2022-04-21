from xgboost import XGBRegressor
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer

def XGB_example():

	# Loading dataset
	data = load_diabetes()

	df = pd.DataFrame(data=data.data, columns=data.feature_names)
	df['Y'] = data.target

	X = df.iloc[: , :-1]
	y = df.iloc[: , -1]

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y,test_size=0.30)

	# Define model
	Model = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3,
                              learning_rate = 0.1, max_depth = 20, alpha = 10, 
                              n_estimators = 40)
	
	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	FIDAP = FeatureImportanceAnalyzer(*args, **kwargs)
	pprint.pprint(FIDAP.get())
	FIDAP.boxplot()