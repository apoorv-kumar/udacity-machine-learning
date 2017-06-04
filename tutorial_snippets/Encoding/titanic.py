import numpy as np
import pandas as pd

X = pd.read_csv("titanic_data.csv")
X = X.dropna()
X = X.select_dtypes(include=[object]) #string is an object

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()

# NOTE : single column index returns Series
#        multiple columns returns sliced DataFrame
# We use dataframe here to make it clear to fit_transform
# that it's a list of single feature and not single element
# of n features
# https://stackoverflow.com/questions/43475439/python-deprecationwarning-passing-1d-arrays-as-data-is-deprecated-in-0-17-and-w

print(type(X['Name']))   #series
print(type(X[['Name']])) #dataframe

for feature in X:
    X[feature] = le.fit_transform(X[[feature]])

print(X)

ohe = OneHotEncoder()

onehotlables = ohe.fit_transform(X)

print(onehotlables)