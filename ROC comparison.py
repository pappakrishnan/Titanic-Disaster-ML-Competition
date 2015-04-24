# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:38:35 2015

@author: venki_k07
"""

#Comparison of various methods using ROC curves
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

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
# Done with Data Munging

#Data modeling
train_survival = train_data[['Survived']]
train_data = con1[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
test_data = con2[['Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked']]
 
#Implementing Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, oob_score=True)

train_x, valid_x, train_y, valid_y = cv.train_test_split(train_data,\
                             train_survival, test_size = 0.3, random_state = 4)
"""
#Model:1 (normal model)
classifier1 = rfc.fit(train_x, np.ravel(train_y))
score_basic = classifier1.score(valid_x, np.ravel(valid_y))

#Model:2 (Grid search)
param_grid_all = {"max_depth": [3, None],
                  "max_features": [3, 4, 5, 7],
                  "min_samples_split": [1, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "bootstrap": [True],
                  "criterion": ["gini", "entropy"]}

crv = cv.KFold(len(train_x), n_folds = 10, indices=True, shuffle=True, random_state=2)
#crv = cv.ShuffleSplit(len(train_x), n_iter = 10, test_size = 0.2, random_state =2)
classifier2 = GridSearchCV(estimator = rfc, cv = crv, param_grid = param_grid_all, n_jobs=-1)
classifier2.fit(train_x, np.ravel(train_y))
#print(classifier2.grid_scores_)
score_cv = classifier2.score(valid_x, np.ravel(valid_y))

#Model:3 (PCA)
variance_pct = 0.99 #Minimum percentage of variance we want to be described by the resulting transformed variables
pca = PCA(n_components = variance_pct)
pca = PCA(n_components = 'mle')

x_data = pca.fit_transform(train_data, np.ravel(train_data))
x_data = pd.DataFrame(x_data)

train_xpca, valid_xpca, train_ypca, valid_ypca = cv.train_test_split(x_data,\
                             train_survival, test_size = 0.3, random_state = 4)
                             
param_grid_pca = {"max_depth": [3, None],
                  "max_features": [2, 4, 6],
                  "min_samples_split": [1, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "bootstrap": [True],
                  "criterion": ["gini", "entropy"]}

crv_pca = cv.KFold(len(train_xpca), n_folds = 10, indices=True, shuffle=True, random_state=2)
classifier3 = GridSearchCV(estimator = rfc, cv = crv_pca, param_grid = param_grid_pca, n_jobs=-1)
classifier3.fit(train_xpca, np.ravel(train_ypca))
#print(classifier3.grid_scores_)
score_cv_pca = classifier3.score(valid_xpca, np.ravel(valid_ypca))            

#ROC curves
fpr1, tpr1, _ = roc_curve(valid_y, classifier1.predict_proba(valid_x)[:,1])
fpr2, tpr2, _ = roc_curve(valid_y, classifier2.predict_proba(valid_x)[:,1])
fpr3, tpr3, _ = roc_curve(valid_ypca, classifier3.predict_proba(valid_xpca)[:,1])

#Calculate Area under the curve (AUC)
ROC_auc1 = auc(fpr1, tpr1)
ROC_auc2 = auc(fpr2, tpr2)
ROC_auc3 = auc(fpr3, tpr3)

print('ROC AUC1:', ROC_auc1)
print('ROC AUC2:', ROC_auc2)
print('ROC AUC3:', ROC_auc3)

#Plotting ROC curve:
plt.figure()
plt.plot(fpr1, tpr1, label ='RF (Area = %0.2f)'% ROC_auc1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr2, tpr2, label ='RF-Grid (Area = %0.2f)'% ROC_auc2)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr3, tpr3, label ='RF-Grid-PCA (Area = %0.2f)'% ROC_auc3)
plt.plot([0,1],[0,1],'k--')

plt.title('ROC Curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc = "lower right")
plt.show()
"""


