import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# iris dataset
path = "iris.data"

# column names
headernames = ["sepal-length", "sepal-width", "petal-length", "petal-width", "Class"]

# read dataset
dataset = pd.read_csv(path, names = headernames)
dataset.head()

# pre process
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# creating train and test datasets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.30)
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)

# outputs
report = classification_report(y_test, y_pred)
print ("Classification report", report)

accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy score", accuracy)

# calculating feature importance

n_features = X_train.shape[1]
n_simulations = 10

feature_importances_ = []

for i in range (n_features):
    X_temp = X_test.copy()
    temp_accuracy_list = []
    for j in range (n_simulations):
        np.random.shuffle(X_temp[:,0])
        y_pred = classifier.predict(X_temp)
        accuracy_temp = accuracy - accuracy_score(y_test, y_pred)
        temp_accuracy_list.append(accuracy_temp)
    feature_importances_.append(abs(round(np.mean(temp_accuracy_list), 4)) if np.mean(temp_accuracy_list)<0 else 0)

print (feature_importances_)