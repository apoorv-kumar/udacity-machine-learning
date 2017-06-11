# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
r1 = recall(y_test,clf1.predict(X_test))
p1 = precision(y_test,clf1.predict(X_test))

clf2 = GaussianNB()
clf2.fit(X, y)
r2 = recall(y_test,clf2.predict(X_test))
p2 = precision(y_test,clf2.predict(X_test))

results = {
  "Naive Bayes Recall": r2,
  "Naive Bayes Precision": p2,
  "Decision Tree Recall": r1,
  "Decision Tree Precision": p1
}
from pprint import pprint
pprint(results)