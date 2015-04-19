# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 17:37:21 2015

@author: Venkatesh
"""
#applying random forest to determine the age
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV

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

#################### train data cleaning ################
train_data_comp = train_data.dropna()
train_comp = train_data_comp.loc[:, ('Pclass','Sex', 'SibSp', 'Parch','Fare','Embarked')]
train_comp_age = train_data_comp.loc[:, ('Age')]

train_data_incomp = train_data.loc[train_data.Age.isnull(),:]
train_data_test = train_data_incomp.loc[:, ('Pclass','Sex', 'SibSp', 'Parch','Fare','Embarked')]

#Implementing Random Forest Classifier
rfr = RandomForestClassifier(n_estimators = 100, n_jobs = -1, max_features = 6, min_samples_split = 1)

age_fit = rfr.fit(train_comp, train_comp_age)
age_predict = age_fit.predict(train_data_test)

age_predict = pd.DataFrame(data = age_predict, columns = ['Age'], index = train_data_test.index.values)  

for i in train_data_test.index.values:
    train_data.loc[i, 'Age'] = age_predict.loc[i, 'Age']
########################## train Data Cleaned ##########

#################### test data cleaning ################
test_data_incomp = test_data.loc[test_data.Age.isnull(),:]
test_data_test = test_data_incomp.loc[:, ('Pclass','Sex', 'SibSp', 'Parch','Fare','Embarked')]

age_predict_test = age_fit.predict(test_data_test)

age_predict_test = pd.DataFrame(data = age_predict_test, columns = ['Age'], index = test_data_test.index.values)  

for i in test_data_test.index.values:
    test_data.loc[i, 'Age'] = age_predict_test.loc[i, 'Age']
########################## test Data Cleaned ##########
    
#splittng the parameters and the target
train_dat = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]
train_survival= train_data['Survived']
 
#Implementing Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 1000,\
    n_jobs = -1, max_features = 7, max_depth = None, min_samples_split = 1, oob_score=True)

#splitting the data in to training data and validation data
train_x, valid_x, train_y, valid_y = cv.train_test_split(train_dat,\
                             train_survival, test_size=0.2, random_state=0)

crv = cv.KFold(len(train_x), n_folds=10, indices=True, shuffle=True, random_state=4)

param_grid_all = {"max_depth": [3, None],
                  "max_features": [3, 4, 5, 6],
                  "min_samples_split": [2, 3, 6],
                  "min_samples_leaf": [1, 3, 6],
                  "criterion": ["gini", "entropy"]}

classifier = GridSearchCV(estimator = rfc, cv = crv, param_grid = param_grid_all, n_jobs=1)

classi = classifier.fit(train_x, train_y)  #works
score_cv = classi.score(valid_x, valid_y)
print("Cross validation score =",score_cv*100,"%")

survival_fit = classifier.fit(train_dat, train_survival)
survival_pred = survival_fit.predict(test_data)
 
survival_pred = pd.Series(survival_pred)
s = {'PassengerId': Id, 'Survived': survival_pred}    
survival_predict = pd.DataFrame(data = s, columns = ['PassengerId','Survived'])
  
survival_predict.to_csv('E:/Kaggle_Titanic/results_03_27_1.csv', index=False)
