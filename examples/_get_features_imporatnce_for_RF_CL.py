def _get_features_imporatnce_for_RF_CL(n_simulations, n_features, X_test, y_test, accuracy, model):
	feature_importances_ = []
	for i in range (n_features):
		X_temp = X_test.copy()
		temp_accuracy_list = []
		for j in range (n_simulations):
			np.random.shuffle(X_temp[:,0])
			y_pred = classifier.predict(X_temp)
			accuracy_temp = accuracy - accuracy_score(y_test, y_pred)
			temp_accuracy_list.append(accuracy_temp)
		feature_importances_.append(abs(round(np.mean(temp_accuracy_list), 4)) if np.mean(temp_accuracy_list)<0 else 0)
	print (feature_importances_)