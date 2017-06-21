from sklearn import svm, grid_search, datasets
iris = datasets.load_iris()

parameters = {'kernel': ('linear', 'rbf'), 'C' :[1,5,10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)

print(clf.best_params_)