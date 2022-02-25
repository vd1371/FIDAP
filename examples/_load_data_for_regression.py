import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def _load_data_for_regression():

	data = load_diabetes()

	df = pd.DataFrame(data=data.data, columns=data.feature_names)
	df['Y'] = data.target

	X = df.iloc[: , :-1].values
	y = df.iloc[: , -1].values

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y,test_size=0.30)

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":

	_load_data_for_regression()

