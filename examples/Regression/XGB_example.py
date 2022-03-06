from xgboost import XGBRegressor
from ._load_data_for_regression import _load_data_for_regression
from ._analyze import _analyze

def XGB_example():

	# Loading dataset
	X_train, X_test, y_train, y_test = _load_data_for_regression()

	# Define model
	Model = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3,
                              learning_rate = 0.1, max_depth = 20, alpha = 10, 
                              n_estimators = 40)
	
	# Fit the model
	Model.fit(X_train,y_train)

	# Feature importance analysis
	_analyze(Model, X_test, y_test, features = None)