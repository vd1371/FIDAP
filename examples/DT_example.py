import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from ._load_data_for_classification import _load_data_for_classification
from FIIL import FeatureImportanceAnalyzer

def DT_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_classification()

	# Define model
	Model = DecisionTreeClassifier()

	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	fiil = FeatureImportanceAnalyzer(Model,
									X_test,
									y_test,
									features = None)
	print (fiil.get())