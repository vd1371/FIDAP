import numpy as np
import pandas as pd

def _prepare_X_Y_features(model, X, Y, **params):

	features = params.get("features")

	if isinstance(X, pd.DataFrame):
		x_out = X.values
	elif isinstance(X, list):
		x_out = np.array(X)
	else:
		x_out = X

	if isinstance(Y, (pd.DataFrame, pd.Series)):
		y_out = Y.values
	elif isinstance(Y, list):
		y_out = np.array(Y)
	else:
		y_out = Y

	if features == None:
		if isinstance(X, pd.DataFrame):
			features = list(X.columns)
		else:
			features = [f'X{i}' for i in range(len(X[0]))]
	else:
		if len(features) != x_out.shape[1]:
			raise ValueError ("Length of user input features must be equal "\
								"to the length of input columns")

	return x_out, y_out, features