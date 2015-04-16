# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:12:26 2015

@author: venki_k07
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data from Excel files as dataframes
train_data = pd.read_csv("C:/Users/venki_k07/Dropbox/Kaggle_Titanic/test.csv")

Id = train_data['PassengerId']
train_data = train_data.loc[:,('Pclass','Age', 'Sex', 'SibSp', 'Parch','Fare','Embarked')]

#mapping Embarked Column
train_data = train_data.replace(['male','female'],[1,0])

#mapping Embarked Column
#Creating port Dictionary
Ports = list(enumerate(train_data['Embarked'].unique()))
Ports_dict = { name : i for i, name in Ports } 
train_data.Embarked = train_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)

final = train_data.dropna()

#age_10 = pd.Series(final.loc[final.Age <=10, 'Fare']).values
#age_20 = pd.Series(final.loc[(final.Age > 10) & (final.Age <= 20), 'Fare']).values
#age_30 = pd.Series(final.loc[(final.Age > 20) & (final.Age <= 30), 'Fare']).values
#age_40 = pd.Series(final.loc[(final.Age > 30) & (final.Age <= 40), 'Fare']).values
#
#fig = plt.figure()
#ax1 = fig.add_subplot(2,2,1)
#ax2 = fig.add_subplot(2,2,2)
#ax3 = fig.add_subplot(2,2,3)
#ax4 = fig.add_subplot(2,2,4)
#
#ax1.hist(age_10)
#ax2.hist(age_20)
#ax3.hist(age_30)
#ax4.hist(age_40)
#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
#plt.title('Age Vs Ticket Fare')
#plt.xlabel('Age')
#plt.ylabel('Ticket Fare')
ax1.scatter(final.ix[(final.Pclass == 1), 'Age'],final.ix[(final.Pclass == 1), 'Fare'])
ax2.scatter(final.ix[(final.Pclass == 2), 'Age'],final.ix[(final.Pclass == 2), 'Fare'])
ax3.scatter(final.ix[(final.Pclass == 3), 'Age'],final.ix[(final.Pclass == 3), 'Fare'])
plt.show()

plt.figure()
#plt.xlim(0,320)
bins1 = np.arange(0, 300, 10)
bins2 = np.arange(0, 80, 5)
bins3 = np.arange(0, 50, 4)
fare1 = pd.Series(final.ix[(final.Pclass == 1), 'Fare']).values
plt.hist(fare1, bins1)
plt.show()
fare1 = pd.Series(final.ix[(final.Pclass == 2), 'Fare']).values
plt.hist(fare1, bins2)
plt.show()
fare1 = pd.Series(final.ix[(final.Pclass == 3), 'Fare']).values
plt.hist(fare1, bins3)
plt.show()
#plt.figure()
#plt.title('Pclass Vs Ticket Fare')
#plt.xlabel('Pclass')
#plt.ylabel('Ticket Fare')
#plt.ylim(0,300)
#plt.scatter(final['Pclass'],final['Fare'])
#plt.show()
