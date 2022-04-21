from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer

def ETR_example():

	# Loading dataset
	data = load_diabetes()

	df = pd.DataFrame(data=data.data, columns=data.feature_names)
	df['Y'] = data.target

	X = df.iloc[: , :-1]
	y = df.iloc[: , -1]

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y,test_size=0.30)

	# Define model
	Model = ExtraTreesRegressor()

	# Fit the model
	Model.fit(X_train, y_train)

	# Feature importance analysis
	FIDAP = FeatureImportanceAnalyzer(Model,
                                    X_test,
                                    y_test,
                                    n_feature_combination = 2,
                                    n_simulations = 20)
	print (FIDAP.get())
	FIDAP.boxplot()