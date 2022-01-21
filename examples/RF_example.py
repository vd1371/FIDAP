from FIIL import FeatureImportanceAnalyzer

def RF_example():
	'''
	TO AMIN: These codes are just for your reference
	Please update as you wish
	'''

	data = load_some_data()
	X, Y = seperate_data(data)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

	model = RandomForestClassifier()
	model.fit(X_train, Y_train)

	ft = FeatureImportanceAnalyzer(model, X_test, Y_test)
	print (ft.get())

