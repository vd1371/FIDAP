import numpy as np
import pandas as pd

def _prepare_X_Y_features(model, X, Y, **params):

	features = params.get("features")

	if isinstance(X, pd.DataFrame):
		x_out = X.values
	elif isinstance(X, list):
		x_out = np.array(X)
	elif isinstance(X, np.ndarray):
		x_out = X
	else:
		raise TypeError ("X must be pd.DataFrame, " + \
							"list of lists, or numpy.ndarray")

	if isinstance(Y, (pd.DataFrame, pd.Series)):
		y_out = Y.values
	elif isinstance(Y, list):
		y_out = np.array(Y)
	elif isinstance(Y, np.ndarray):
		y_out = Y.reshape(-1)
	else:
		raise TypeError("Y must be pd.DataFrame, pd.Series, 1D series, or " +\
							"1d numpy array")

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