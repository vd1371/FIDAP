from numpy import sqrt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# load the dataset
file = pd.read_csv("Boston.csv")
file_to_matrix = np.asfarray(file)
file_to_matrix_WO_first_column = np.delete(file_to_matrix, 0, 1)

# split into input and output columns
X = file_to_matrix_WO_first_column[:, :-1]
y = file_to_matrix_WO_first_column[:, -1]

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mse')

# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

# make a prediction
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)

# calculating feature importance
n_simulations = 10
feature_importances_ = []
for i in range (n_features):
    X_temp = X_test.copy()
    temp_error_list = []
    for j in range (n_simulations):
        np.random.shuffle(X_temp[:,0])
        error_temp = error - model.evaluate(X_temp, y_test, verbose=0)
        temp_error_list.append(error_temp)
    feature_importances_.append(abs(round(np.mean(temp_error_list), 4)) if np.mean(temp_error_list)<0 else 0)

print (feature_importances_)