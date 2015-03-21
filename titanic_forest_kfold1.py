# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 04:55:15 2014

@author: Venkatesh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV

#Reading data from Excel files as dataframes
train_data = pd.read_csv("E:/Kaggle_Titanic/train.csv")
test_data = pd.read_csv("E:/Kaggle_Titanic/test.csv")

Id = test_data['PassengerId']
train_data_survival = train_data.loc[:,('Survived')]
train_data = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]
train_data = train_data.replace(['male','female'],[1,0])
test_data = test_data.loc[:,('Pclass', 'Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]
test_data = test_data.replace(['male','female'],[1,0])

#Handling missing data
#Replacing missing age with median age
#Train data
median_age = train_data['Age'].dropna().median()
if len(train_data.Age[ train_data.Age.isnull() ]) > 0:
    train_data.loc[ (train_data.Age.isnull()), 'Age'] = median_age

#cleaning Embarked Column
#Creating port Dictionary
Ports = list(enumerate(train_data['Embarked'].unique()))
Ports_dict = { name : i for i, name in Ports }     
# replacing for train data
if len(train_data.Embarked[ train_data.Embarked.isnull() ]) > 0:
    train_data.Embarked[ train_data.Embarked.isnull() ] = train_data.Embarked.dropna().mode().values
# replacing for test data
if len(test_data.Embarked[ test_data.Embarked.isnull() ]) > 0:
    test_data.Embarked[ test_data.Embarked.isnull() ] = test_data.Embarked.dropna().mode().values
    
# set up a dictionary in the form  Ports : index  Replaces char to int from 0 to n
train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int) 
test_data.Embarked = test_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#Test data
median_age_test = test_data['Age'].dropna().median()
if len(test_data.Age[ test_data.Age.isnull() ]) > 0:
    test_data.loc[ (test_data.Age.isnull()), 'Age'] = median_age_test
#Test Fare
if len(test_data.Fare[ test_data.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_data[ test_data.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_data.loc[ (test_data.Fare.isnull()) & (test_data.Pclass == f+1 ), 'Fare'] = median_fare[f]

#Implementing Random Forest Classifier
rfr = RandomForestClassifier(n_estimators = 1000,\
    n_jobs = -1, max_features = 7, max_depth = None, min_samples_split = 1, oob_score=True)

#splitting the data in to training data and validation data
train_x, valid_x, train_y, valid_y = cv.train_test_split(train_data,\
                             train_data_survival, test_size=0.2, random_state=0)

#crv = cv.ShuffleSplit(len(train_data), n_iter = 10, test_size = 0.2, random_state =2) #Cross validation strategy
crv = cv.KFold(len(train_x), n_folds=10, indices=True, shuffle=True, random_state=4)

#==============================================================================
# param_grid_all = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [1, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
#==============================================================================
param_grid_all = {"max_depth": [3, None],
                  "max_features": [3, 5, 7], 
                  "criterion": ["gini", "entropy"]}
classifier = GridSearchCV(estimator = rfr, cv = crv, param_grid = param_grid_all, n_jobs=1)

try:
    score_cv = cv.cross_val_score(classifier, train_x, train_y, cv = crv, n_jobs=1)
except IndexError:
    score_cv = "Index Error"

#try:
#    score_cv = [classifier.fit(train_x[train], train_y[train]).score(train_x[test], train_y[test]) for train, test in crv]
#except IndexError:
#    score_cv = "Index Error"

#classi = classifier.fit(train_x, train_y)  #works
#score_cv = classi.score(valid_x, valid_y)
print(score_cv)

#==============================================================================
# #fitting for the whole dataset
# survival_fit = classifier.fit(train_data, train_data_survival)
# survival_pred = survival_fit.predict(test_data)
# 
# survival_pred = pd.Series(survival_pred)
# s = {'PassengerId': Id, 'Survived': survival_pred}    
# survival_predict = pd.DataFrame(data = s, columns = ['PassengerId','Survived'])
#  
# survival_predict.to_csv('E:/Kaggle_Titanic/results_12_19.csv', index=False)
#==============================================================================

