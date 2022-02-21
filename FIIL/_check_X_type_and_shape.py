import pandas as pd
import numpy as np

def _check_X_Y_type_and_shape(X, Y):

	if not isinstance(X, (pd.DataFrame, np.ndarray, list)):
		raise TypeError ("Type of X MUST be a list of lists, ",
							"pandas DataFrame, or numpy 2d array")

	if isinstance(X, np.ndarray) and not X.ndim == 2:
		raise TypeError ("if X is passed as a numpy array, it MUST be 2d")

	elif isinstance(X, list) and not np.array(X).ndim == 2:
		raise TypeError ("if X is passed as a list, it MUST be list of lists")
