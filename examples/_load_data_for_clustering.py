import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def _load_data_for_clustering():

	data, true_labels = make_blobs(n_samples=[300,350,250], 
                                   n_features=4, random_state=105)

	df = pd.DataFrame(data=data.data, columns=data.feature_names)

	X = df.iloc[: , :].values

	return X


if __name__ == "__main__":

	_load_data_for_clustering()

