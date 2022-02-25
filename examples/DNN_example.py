import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from ._load_data_for_regression import _load_data_for_regression
from FIIL import FeatureImportanceAnalyzer

def DNN_example():

    # Loading dataset
    X_train, X_test, y_train, y_test = _load_data_for_regression()
    
    # Determining the number of input features
    n_features = X_train.shape[1]

    # Define model
    Model = Sequential()

    Model.add(Dense(10, activation='relu', kernel_initializer='he_normal', 
                    input_shape=(n_features,)))
    Model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    Model.add(Dense(1))

    # Compile the model
    Model.compile(optimizer='adam', loss='mse')

    # Fit the model
    Model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # Feature importance analysis
    fiil = FeatureImportanceAnalyzer(Model,
                                    X_test,
                                    y_test,
                                    features = None)
    print (fiil.get())