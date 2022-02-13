import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from FIIL import FeatureImportanceAnalyzer


def DNN_example():

    # Loading Boston Dataset
    path = "https://raw.githubusercontent.com/selva86/datasets/" \
            "master/BostonHousing.csv"
    file = pd.read_csv(path)

    X = file.iloc[: , :-1].values
    y = file.iloc[: , -1].values

    #X, y = load_boston(return_X_y=True)

    # Creating train and test datasets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Determine the number of input features
    n_features = X_train.shape[1]

    # Define model
    classifier = Sequential()
    classifier.add(Dense(10, activation='relu', kernel_initializer='he_normal', 
                    input_shape=(n_features,)))
    classifier.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    classifier.add(Dense(1))

    # Compile the model
    classifier.compile(optimizer='adam', loss='mse')

    # Fit the model
    classifier.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # Evaluate the model
    error = classifier.evaluate(X_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))

    # Making a prediction
    #row = [0.00632,18.00,2.310,0,0.5380,6.5750,
           #65.20,4.0900,1,296.0,15.30,396.90,4.98]
    #yhat = classifier.predict([row])
    #print('Predicted: %.3f' % yhat)

    # Calculating feature importance
    n_simulations = 10

    fiil = FeatureImportanceAnalyzer(classifier, file)
    print (fiil.get())