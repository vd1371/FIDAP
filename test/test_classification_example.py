import unittest

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from catboost import CatBoostClassifier

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer

class TestClassification(unittest.TestCase):

	def test_classification(self):

		for classification_model in [
			SVC, RandomForestClassifier,
			PassiveAggressiveClassifier,
			LogisticRegression,
			ExtraTreesClassifier,
			DecisionTreeClassifier,
			GradientBoostingClassifier,
			MLPClassifier,
			CatBoostClassifier,
			# RadiusNeighborsClassifier
			]:



			# Loading dataset
			data = load_iris()

			df = pd.DataFrame(data=data.data, columns=data.feature_names)
			df['Y'] = data.target

			X = df.iloc[: , :-1]
			y = df.iloc[: , -1]

			X_train, X_test, y_train, y_test = \
				train_test_split(X, y,test_size=0.30)

			# Define model
			model = classification_model()

			# Fit the model
			model.fit(X_train.values, y_train.values)

			# Feature importance analysis
			FIDAP = FeatureImportanceAnalyzer(model, X_train, y_train)
			FIDAP.get(verbose=False)
			FIDAP.boxplot()