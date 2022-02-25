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


	if isinstance(X, pd.DataFrame):
		features = list(X.columns)
	else:
		features = [f'X{i}' for i in range(len(X[0]))]

	return x_out, y_out, features