

def get_features_importance(**params):

	model = params.get("model")

	if type(model) == "RandomForestsClassifier":
		return _get_features_imporatnce_for_RF_CL(**params)