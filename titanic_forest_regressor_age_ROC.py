# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:56:59 2015

@author: venki_k07
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

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

#Test Fare (mode)
if len(test_data.Fare[test_data.Fare.isnull()]) > 0:
    mean_fare = np.zeros(3)
    mean_fare[0] = test_data[(test_data.Pclass == 1) & ((test_data.Fare <= 170) & (test_data.Fare >= 20))]['Fare'].dropna().mean()
    mean_fare[1] = test_data[(test_data.Pclass == 2) & ((test_data.Fare <= 40) & (test_data.Fare >= 10))]['Fare'].dropna().mean()
    mean_fare[2] = test_data[(test_data.Pclass == 3) & ((test_data.Fare <= 25) & (test_data.Fare >= 4))]['Fare'].dropna().mean()
    for f in range(0,3):
        test_data.loc[ (test_data.Fare.isnull()) & (test_data.Pclass == f+1 ), 'Fare'] = mean_fare[f]

#Applying linear regression to impute Age and Fare
clf = linear_model.LinearRegression()

#Imputing age using a linear Regressor
con1 = train_data[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
con2 = test_data[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
    
con1_null = con1[con1.Age.isnull()]
con2_null = con2[con2.Age.isnull()]
con1_notnull = con1[con1.Age.notnull()]
    
data = con1_notnull
Y = data.values[:, 0]
X = data.values[:, 1::]
    
clf.fit(X, Y)

con1_ages = clf.predict(con1_null.values[:, 1::])
con2_ages = clf.predict(con2_null.values[:, 1::])

con1.loc[(con1.Age.isnull()), 'Age'] = con1_ages
con2.loc[(con2.Age.isnull()), 'Age'] = con2_ages

train_survival = train_data[['Survived']]
train_data = con1[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
test_data = con2[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
 
#Implementing Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, oob_score=True)

train_x, valid_x, train_y, valid_y = cv.train_test_split(train_data,\
                             train_survival, test_size = 0.1, random_state = 4)
                             
rfc.fit(train_x, train_y)

fpr, tpr, _ = roc_curve(valid_y, rfc.predict_proba(valid_x)[:,1])

#Calculate Area under the curve (AUC)
ROC_auc = auc(fpr, tpr)
print('ROC AUC:', ROC_auc)

#Plotting ROC curve:
plt.figure()
plt.plot(fpr, tpr, label ='ROC curve (Area = %0.2f)'% ROC_auc)
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc = "lower right")
plt.show()

                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             