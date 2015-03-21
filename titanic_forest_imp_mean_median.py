# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:31:17 2015

@author: Venkatesh
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
#import pylab as py
#import matplotlib.pyplot as plt

#Reading data from Excel files as dataframes
train_data = pd.read_csv("E:/Kaggle_Titanic/train.csv")
test_data = pd.read_csv("E:/Kaggle_Titanic/test.csv")

Id = test_data['PassengerId']
train_data = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked', 'Survived')]
test_data = test_data.loc[:,('Pclass', 'Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]

#mapping Embarked Column
train_data = train_data.replace(['male','female'],[1,0])
test_data = test_data.replace(['male','female'],[1,0])

#mapping Embarked Column
#Creating port Dictionary
Ports = list(enumerate(train_data['Embarked'].unique()))
Ports_dict = { name : i for i, name in Ports } 
train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int) 
test_data.Embarked = test_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#Handling missing data
#Replacing missing age with median age 
#Train data
missing_columns = []
for type in train_data.columns.values:
    length_missing_data = len(train_data.loc[train_data.loc[:,type].isnull(), type])
    if length_missing_data > 0:
        missing_columns.append(type)
if len(missing_columns)>0:
    print('Columns that needs imputation in training data', missing_columns)
else:
    print('No missing values in training data')
#test data
missing_columns_test = []
for type in test_data.columns.values:
    length_missing_data = len(test_data.loc[test_data.loc[:,type].isnull(), type])
    if length_missing_data > 0:
        missing_columns_test.append(type)
if len(missing_columns_test)>0:
    print('Columns that needs imputation in testing data', missing_columns_test)
else:
    print('No missing values in testing data')
    
#Test Fare
if len(test_data.Fare[test_data.Fare.isnull()]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_data[ test_data.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_data.loc[ (test_data.Fare.isnull()) & (test_data.Pclass == f+1 ), 'Fare'] = median_fare[f]
        
#imputing age column training data
parch = train_data.ix[:, 'Parch'].dropna().unique()
SibSp = train_data.ix[:, 'SibSp'].dropna().unique()

mean = train_data.loc[:,'Age'].dropna().mean()
for i in parch:
    for j in SibSp:
        mod = train_data.loc[(train_data.Parch==i) & (train_data.SibSp==j),'Age'].dropna().mode()
        if len(mod)>1:
            train_data.loc[(train_data.Age.isnull()) & (train_data.Parch==i) & (train_data.SibSp==j), 'Age'] = mod.mean()
        elif len(mod)==1:
            train_data.loc[(train_data.Age.isnull()) & (train_data.Parch==i) & (train_data.SibSp==j), 'Age'] = mod
        else:
            train_data.loc[(train_data.Age.isnull()) & (train_data.Parch==i) & (train_data.SibSp==j), 'Age'] = mean
            
train_data.loc[(train_data.Age.isnull()), 'Age'] = mean        

#imputing age column testing data
parch = test_data.ix[:, 'Parch'].dropna().unique()
SibSp = test_data.ix[:, 'SibSp'].dropna().unique()

mean = test_data.loc[:,'Age'].dropna().mean()
for i in parch:
    for j in SibSp:
        mod = test_data.loc[(test_data.Parch==i) & (test_data.SibSp==j),'Age'].dropna().mode()
        if len(mod)>1:
            test_data.loc[(test_data.Age.isnull()) & (test_data.Parch==i) & (test_data.SibSp==j), 'Age'] = mod.mean()
        elif len(mod)==1:
            test_data.loc[(test_data.Age.isnull()) & (test_data.Parch==i) & (test_data.SibSp==j), 'Age'] = mod
        else:
            test_data.loc[(test_data.Age.isnull()) & (test_data.Parch==i) & (test_data.SibSp==j), 'Age'] = mean
            
test_data.loc[(test_data.Age.isnull()), 'Age'] = mean 

#new_train_data = train_data.dropna()

#splittng the parameters and the target
train_dat = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]
train_survival= train_data['Survived']
 
#Implementing Random Forest Classifier
rfr = RandomForestClassifier(n_estimators = 1000,\
    n_jobs = -1, max_features = 7, max_depth = None, min_samples_split = 1, oob_score=True)

#rfr.fit(train_dat,train_survival)
#fi = enumerate(rfr.feature_importances_)  #provides the importance of every feature
#cols = train_dat.columns     #if any of the feature is of less importance it can be removed from the Randomforest training
#print([(value,cols[i]) for (i,value) in fi])

#splitting the data in to training data and validation data
train_x, valid_x, train_y, valid_y = cv.train_test_split(train_dat,\
                             train_survival, test_size=0.2, random_state=0)

#crv = cv.ShuffleSplit(len(train_data), n_iter = 10, test_size = 0.2, random_state =2) #Cross validation strategy
crv = cv.KFold(len(train_x), n_folds=10, indices=True, shuffle=True, random_state=4)

#param_grid_all = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [1, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}

param_grid_all = {"max_depth": [3, None],
                  "max_features": [3, 4, 5, 6],
                  "min_samples_split": [2, 3, 6],
                  "min_samples_leaf": [1, 3, 6],
                  "criterion": ["gini", "entropy"]}

classifier = GridSearchCV(estimator = rfr, cv = crv, param_grid = param_grid_all, n_jobs=1)

#try:
#    score_cv = cv.cross_val_score(classifier, train_x, train_y, cv = crv, n_jobs=1)
#except IndexError:
#    score_cv = "Index Error"

classi = classifier.fit(train_x, train_y)  #works
score_cv = classi.score(valid_x, valid_y)
print(score_cv)

survival_fit = classifier.fit(train_dat, train_survival)
survival_pred = survival_fit.predict(test_data)
 
survival_pred = pd.Series(survival_pred)
s = {'PassengerId': Id, 'Survived': survival_pred}    
survival_predict = pd.DataFrame(data = s, columns = ['PassengerId','Survived'])
  
survival_predict.to_csv('E:/Kaggle_Titanic/results_01_31.csv', index=False)