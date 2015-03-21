# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:55:46 2015

@author: Venkatesh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV

#Reading data from Excel files as dataframes
train = pd.read_csv("E:/Kaggle_Titanic/train.csv")
test = pd.read_csv("E:/Kaggle_Titanic/test.csv")

Id = test['PassengerId']
train = train.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked', 'Survived')]
test = test.loc[:,('Pclass', 'Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]

#mapping sex Column
train = train.replace(['male','female'],[1,0])
test = test.replace(['male','female'],[1,0])

#mapping Embarked Column
train = train.replace(['S', 'C', 'Q'], [0, 1, 2])
test = test.replace(['S', 'C', 'Q'], [0, 1, 2])

#Finding columns with missing values
#Train and test data
for dat in (train, test):
    miss_cols = []
    for type in dat.columns.values:
        len_miss_data = len(dat.loc[dat.loc[:,type].isnull(), type])
        if len_miss_data > 0:
            miss_cols.append(type)
    if len(miss_cols) > 0:
        print('Columns that needs imputation are', miss_cols)
    else:
        print('No missing values in the data')

#Replacing missing values
#Train Embarked
train.Embarked[train.Embarked.isnull()] = train.Embarked.dropna().mode().values

for i in test.groupby('Pclass')['Fare'].mean().index.values:
    test.loc[ (test.Fare.isnull()) & (test.Pclass == i), 'Fare'] = test.groupby('Pclass')['Fare'].mean()[i]

for j in train['Pclass'].unique():
    median_age = train[ train.Pclass == j ]['Age'].dropna().median()
    train.loc[ (train.Age.isnull()) & (train.Pclass == j), 'Age'] = median_age
    
for k in test['Pclass'].unique():
    median_age = test[ test.Pclass == k ]['Age'].dropna().median()
    test.loc[ (test.Age.isnull()) & (test.Pclass == k), 'Age'] = median_age

train_x = train.loc[:, ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')]
train_y = train['Survived']

train_x, test_x, train_y, test_y = cv.train_test_split(train_x, train_y, test_size=0.2, random_state=2)
                             
rfc = RandomForestClassifier(n_estimators = 200, n_jobs = 1, max_features = 7, min_samples_split = 1)

train_x = pd.DataFrame(data = train_x, columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
test_x = pd.DataFrame(data = test_x, columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

rfc.fit(train_x, train_y)
fi = enumerate(rfc.feature_importances_)  #provides the importance of every feature
cols = train_x.columns     #if any of the feature is of less importance it can be removed from the Randomforest training
print([(value,cols[i]) for (i,value) in fi])

#Based on the importances we pick the variables to be considered
#train_x = np.array( train_x.ix[:, ('Pclass', 'Sex', 'Age', 'Fare', 'SibSp')] )
#test_x = np.array( test_x.ix[:, ('Pclass', 'Sex', 'Age', 'Fare', 'SibSp')] )
train_x = np.array( train_x )
test_x = np.array( test_x )

rfc_new = RandomForestClassifier(n_estimators = 1000, n_jobs = 1, min_samples_split = 1)

crv = cv.KFold(len(train_x), n_folds=10, indices=True, shuffle=True, random_state = 4)

param_grid_all = {"max_depth": [3, None],
                  "max_features": [5, 6, 7], 
                  "criterion": ["gini", "entropy"]}
                  
classifier = GridSearchCV(estimator = rfc_new, cv = crv, param_grid = param_grid_all, n_jobs=1)

cv_score = cv.cross_val_score(rfc_new, train_x, train_y, cv=crv, n_jobs=1)
print(cv_score.mean())

#Index Issue in the following cv method
#result = [classifier.fit(train_x[train_indices], train_y[train_indices]).score(train_x[test_indices], train_y[test_indices])\
#for train_indices, test_indices in crv]
#print(classifier.fit(train_x, train_y).score(train_x, train_y))



