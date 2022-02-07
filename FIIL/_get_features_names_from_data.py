

def _get_features_names_from_data(X):

	'''
	If X is pandas dataframe, return X.columns as list
	elif numpy ndarray: [f"X{i}" for i in range(X.shape[1])]
	elif list of lists: [f"X{i}" for i in range(len(X))]
	'''

	return features_names