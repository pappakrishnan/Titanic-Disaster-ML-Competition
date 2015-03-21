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

#Reading data from Excel files as dataframes
train_data = pd.read_csv("E:/Kaggle_Titanic/train.csv")

Id = train_data['PassengerId']
train_data = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked', 'Survived')]

#mapping Embarked Column
train_data = train_data.replace(['male','female'],[1,0])

#mapping Embarked Column
#Creating port Dictionary
Ports = list(enumerate(train_data['Embarked'].unique()))
Ports_dict = { name : i for i, name in Ports } 
train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int) 

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

train_data_comp = train_data.dropna()
train_comp = train_data_comp.loc[:, ('Pclass','Sex', 'SibSp', 'Parch','Fare','Embarked', 'Survived')]
train_comp_age = train_data_comp.loc[:, ('Age')]

train_data_incomp = train_data.loc[train_data.Age.isnull(),:]
train_data_test = train_data_incomp.loc[:, ('Pclass','Sex', 'SibSp', 'Parch','Fare','Embarked', 'Survived')]

#Implementing Random Forest Classifier
rfr = RandomForestClassifier(n_estimators = 100, n_jobs = -1, max_features = 7, min_samples_split = 1)

age_fit = rfr.fit(train_comp, train_comp_age)
age_predict = age_fit.predict(train_data_test)

age_predict = pd.DataFrame(data = age_predict, columns = ['Age'], index = train_data_test.index.values)  

for i in train_data_test.index.values:
    train_data.loc[i, 'Age'] = age_predict.loc[i, 'Age']

########################## train Data Cleaned #######################

















