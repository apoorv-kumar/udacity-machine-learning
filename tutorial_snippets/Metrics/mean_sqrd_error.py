import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y)
reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)
mse1 = mse(y_test, reg1.predict(X_test))

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
mse2 = mse(y_test, reg2.predict(X_test))

results = {
 "Linear Regression": mse2,
 "Decision Tree": mse1
}

import pprint
pprint.pprint(results)