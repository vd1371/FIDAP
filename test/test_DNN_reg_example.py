import unittest
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from FIDAP import FeatureImportanceAnalyzer

class DNN_example(unittest.TestCase):

    def test_DNN(self):
        # Loading dataset
        data = load_diabetes()

        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        df['Y'] = data.target

        X = df.iloc[: , :-1]
        y = df.iloc[: , -1]

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,test_size=0.30)

        # Determining the number of input features
        n_features = X_train.shape[1]

        # Define model
        model = Sequential()

        model.add(Dense(10, activation='relu', kernel_initializer='he_normal',
                        input_shape=(n_features,)))
        model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Fit the model
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

        # Feature importance analysis
        FIDAP = FeatureImportanceAnalyzer(model, X_train, y_train, modelling_type='regression')
        FIDAP.get(verbose=True)
        FIDAP.boxplot()