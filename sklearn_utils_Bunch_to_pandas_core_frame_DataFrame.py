from sklearn.datasets import load_boston
import pandas as pd


boston = load_boston()
print(type(boston))

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
print(data)