# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:44:19 2015

@author: venki_k07
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV

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

classi = rfc.fit(train_data, np.ravel(train_survival))

#plotting feature importances
feature_importance = rfc.feature_importances_
feature_importance = 100*(feature_importance / feature_importance.sum())

#feature_importance_index = list(enumerate(feature_importance))
sorted_feature_imp = np.argsort(feature_importance)[::-1]

fig = plt.figure()
pos = np.arange(len(train_data.columns)) + 0.5
plt.barh(pos, feature_importance[sorted_feature_imp[::-1]], align = 'center')
plt.yticks(pos, train_data.columns[sorted_feature_imp[::-1]])
plt.xlabel('Relative importance in %')
plt.title('Feature Importances')
plt.show()

#splitting the data in to training data and validation data
train_x, valid_x, train_y, valid_y = cv.train_test_split(train_data,\
                             train_survival, test_size = 0.2, random_state = 4) 

param_grid_all = {"max_depth": [3, None],
                  "max_features": [3, 4, 5, 7],
                  "min_samples_split": [1, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "bootstrap": [True],
                  "criterion": ["gini", "entropy"]}         

"""
crv = cv.KFold(len(train_x), n_folds = 10, indices=True, shuffle=True, random_state=2)
crv = cv.ShuffleSplit(len(train_x), n_iter = 10, test_size = 0.2, random_state =2)
classifier = GridSearchCV(estimator = rfc, cv = crv, param_grid = param_grid_all, n_jobs=-1)
classifier.fit(train_x, np.ravel(train_y))
#print(classifier.grid_scores_)
score_cv = classifier.score(valid_x, valid_y)
print(score_cv)

survival_pred = rfc.predict(test_data)
 
survival_pred = pd.Series(survival_pred)
s = {'PassengerId': Id, 'Survived': survival_pred}    
survival_predict = pd.DataFrame(data = s, columns = ['PassengerId','Survived'])
  
survival_predict.to_csv('C:/Users/venki_k07/Dropbox/Kaggle_Titanic/results_03_29.csv', index=False)  
"""

# ------ Not working with GridsearchCV -------- #
#score_cv = [classifier.fit(train_x[train], np.ravel(train_y[train])).score(train_x[test], np.ravel(train_y[test])) for train, test in crv]
#cv_score = cv.cross_val_score(rfc, train_x, np.ravel(train_y), cv=5, n_jobs=-1)
#print(cv_score.mean())
# --------------------------------------------- #

#score_cv = classifier.score(valid_x, valid_y)
#print(score_cv)
#
#fi = enumerate(rfc.feature_importances_)
#cols = train_data.columns
#print([(value,cols[i]) for (i,value) in fi]) 