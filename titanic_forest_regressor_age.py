# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:54:29 2015

@author: venki_k07
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation as cv

#Reading data from Excel files as dataframes
train_data = pd.read_csv("C:/Users/venki_k07/Dropbox/Kaggle_Titanic/train.csv")
test_data = pd.read_csv("C:/Users/venki_k07/Dropbox/Kaggle_Titanic/test.csv")

Id = test_data['PassengerId']
train_data = train_data[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked', 'Survived']]
test_data = test_data[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]

#mapping Embarked Column
train_data = train_data.replace(['male','female'],[1,0])
test_data = test_data.replace(['male','female'],[1,0])

#mapping Embarked Column #Creating port Dictionary
Ports = list(enumerate(train_data['Embarked'].unique()))
Ports_dict = { name : i for i, name in Ports }
train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int) 
test_data.Embarked = test_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#Identifying columns with missing data
#Train data
missing_columns = []
for type in train_data.columns.values:
    length_missing_data = len(train_data.loc[train_data.loc[:,type].isnull(), type])
    if length_missing_data > 0:
        missing_columns.append(type)
if len(missing_columns)>0:
    print('Columns that needs imputation in the training data:', missing_columns)
else:
    print('No missing values in the training data')
    
#test data
missing_columns = []
for type in test_data.columns.values:
    length_missing_data = len(test_data.loc[test_data.loc[:,type].isnull(), type])
    if length_missing_data > 0:
        missing_columns.append(type)
if len(missing_columns)>0:
    print('Columns that needs imputation in the testing data:', missing_columns)
else:
    print('No missing values in the testing data')

#Test Fare (mode)
if len(test_data.Fare[test_data.Fare.isnull()]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = test_data[ test_data.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):
        test_data.loc[ (test_data.Fare.isnull()) & (test_data.Pclass == f+1 ), 'Fare'] = median_fare[f]
        
#Imputing age using a random forest Regressor
#concatenating two data frames for building a better model
    con1 = train_data[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
    con2 = test_data[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
    
    con1_null = con1[con1.Age.isnull()]
    con2_null = con2[con2.Age.isnull()]
    
    con1_notnull = con1[con1.Age.notnull()]
    con2_notnull = con2[con2.Age.notnull()]

    
    data = pd.concat([con1_notnull, con2_notnull])
    Y = data.values[:, 0]
    X = data.values[:, 1::]
    
    rf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
    rf.fit(X, Y)
    
    con1_ages = rf.predict(con1_null.values[:, 1::])
    con2_ages = rf.predict(con2_null.values[:, 1::])
    
    con1.loc[(con1.Age.isnull()), 'Age'] = con1_ages
    con2.loc[(con2.Age.isnull()), 'Age'] = con2_ages
    
    train_survival = train_data[['Survived']]
    train_data = con1[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
    test_data = con2[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
        
#Implementing Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, max_features = 7, oob_score=True)

#splitting the data in to training data and validation data
train_x, valid_x, train_y, valid_y = cv.train_test_split(train_data,\
                             train_survival, test_size = 0.1, random_state = 0)        


train_y = np.ravel(train_y)
       
classi = rfc.fit(train_x, train_y) 
score_cv = classi.score(valid_x, valid_y)
print(score_cv)

fi = enumerate(rfc.feature_importances_)
cols = train_data.columns
print([(value,cols[i]) for (i,value) in fi])        
        
#survival_fit = classifier.fit(train_dat, train_survival)
survival_pred = rfc.predict(test_data)
 
survival_pred = pd.Series(survival_pred)
s = {'PassengerId': Id, 'Survived': survival_pred}    
survival_predict = pd.DataFrame(data = s, columns = ['PassengerId','Survived'])
  
survival_predict.to_csv('C:/Users/venki_k07/Dropbox/Kaggle_Titanic/results_03_29.csv', index=False)        
        
        