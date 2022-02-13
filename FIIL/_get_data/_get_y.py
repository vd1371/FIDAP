# dataset has to have a pandas dataframework structure
import pandas as pd

def _get_y(file):
	if not isinstance (file, pd.core.frame.DataFrame):
		raise TypeError("The dataset must have a Pandas dataframe structure")
	else:
		return file.iloc[: , -1].values