# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 04:55:15 2014

@author: Venkatesh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv


#Reading data from Excel files
train_data = pd.read_csv("E:/Kaggle_Titanic/train.csv")
test_data = pd.read_csv("E:/Kaggle_Titanic/test.csv")

Id = test_data['PassengerId']
train_data_survival = train_data.loc[:,('Survived')]
train_data = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]
train_data = train_data.replace(['male','female'],[1,0])
test_data = test_data.loc[:,('Pclass', 'Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]
test_data = test_data.replace(['male','female'],[1,0])

#Handling misssing data
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

#splitting the training data
#crv = cv.ShuffleSplit(len(train_data), n_iter = 10, test_size = 0.2, random_state =2) #Cross validation strategy
crv = cv.KFold(len(train_data), n_folds=10, indices=True, shuffle=True, random_state=4)
score_cv = cv.cross_val_score(rfr, train_data, train_data_survival, cv = crv)  
    
print(score_cv, score_cv.mean())   

survival_fit = rfr.fit(train_data, train_data_survival)
survival_pred = survival_fit.predict(test_data)

#==============================================================================
# survival_fit = rfr.fit(train_data, train_data_survival)
# #provides the importance of every feature
# fi = enumerate(rfr.feature_importances_)
# #if any of the feature is of less importance it could be removed from the training
# cols = train_data.columns
# print([(value,cols[i]) for (i,value) in fi])
# 
# #from the fit it calculates the score for the existing dataset
# #score = survival_fit.score(train_data, train_data_survival)
# score = cv.cross_val_score(rfr, train_data, train_data_survival)
# print("oob_score:", rfr.oob_score_)
# print("Score:", score.mean())
# 
# 
# survival_pred = survival_fit.predict(test_data) #predicting the survival using the fit
# 
# survival_pred = pd.Series(survival_pred)
# 
# s = {'PassengerId': Id, 'Survived': survival_pred}    
# survival_predict = pd.DataFrame(data = s, columns = ['PassengerId','Survived'])
# 
# survival_predict.to_csv('E:/Kaggle_Titanic/results_12_12.csv', index=False)
#==============================================================================
