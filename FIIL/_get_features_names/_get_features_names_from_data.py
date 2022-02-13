# dataset has to have a pandas dataframework structure
def _get_features_names_from_data(file):

	features_names = []

	for feature_name in file.columns:
		features_names.append(feature_name)

	return features_names