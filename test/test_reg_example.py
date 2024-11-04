import unittest
import pprint

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor


import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer

class RegressionTest(unittest.TestCase):

	def test_reg(self):

		for reg_model in [
			LinearRegression,
			DecisionTreeRegressor,
			ExtraTreesRegressor,
			XGBRegressor,
			SVR,
			BayesianRidge,
			PassiveAggressiveRegressor,
			GradientBoostingRegressor
		]:

			# Loading dataset
			data = load_diabetes()

			df = pd.DataFrame(data=data.data, columns=data.feature_names)
			df['Y'] = data.target

			X = df.iloc[: , :-1]
			y = df.iloc[: , -1]

			X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30)

			# Define model
			model = reg_model()

			# Fit the model
			model.fit(X_train.values, y_train.values)

			# Feature importance analysis
			FIDAP = FeatureImportanceAnalyzer(model, X_train, y_train)
			FIDAP.get(verbose=True)
			FIDAP.boxplot()