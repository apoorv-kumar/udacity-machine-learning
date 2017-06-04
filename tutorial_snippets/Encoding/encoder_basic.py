from sklearn import preprocessing
import pandas as pd
sample_data = { 
    'name' : ['Ray', 'Adam', 'Jason', 'Varun'],
    'health' : ['fit', 'slim', 'obese', 'fit']
 }

data = pd.DataFrame(sample_data)

print(data)

label_encoder = preprocessing.LabelEncoder()

label_encoder.fit(data['health'])

# we could do multiple columns at once with pandas get dummies
# see - http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

encoded = label_encoder.transform(data['health'])
print("label encoded:" + str(encoded))

#   health   name
# 0    fit    Ray
# 1   slim   Adam
# 2  obese  Jason
# 3    fit  Varun
# label encoded:[0 2 1 0]

label_encoded_data = label_encoder.fit_transform(data['health'])
# reshape tells fittransform that it's single feature array
shaped_data = label_encoded_data.reshape(-1,1)
print("shaped data:\n" + str(shaped_data))

# shaped data:
# [[0]
#  [2]
#  [1]
#  [0]]

ohe = preprocessing.OneHotEncoder(sparse=False) 
ohe_encoded = ohe.fit_transform(shaped_data)
print("ohe encoded (array representation):\n" + str(ohe_encoded))

# ohe encoded (array representation):
# [[ 1.  0.  0.]
#  [ 0.  0.  1.]
#  [ 0.  1.  0.]
#  [ 1.  0.  0.]]

ohe = preprocessing.OneHotEncoder() 
ohe_encoded = ohe.fit_transform(shaped_data)
print("ohe encoded (sparse matrix representation):\n" + str(ohe_encoded))

# ohe encoded (sparse matrix representation):
#   (0, 0)        1.0
#   (1, 2)        1.0
#   (2, 1)        1.0
#   (3, 0)        1.0