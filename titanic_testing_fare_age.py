# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 03:19:27 2015

@author: Venkatesh
"""

import pandas as pd
#import pylab as py
import matplotlib.pyplot as plt

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

final = train_data.dropna()

age_10 = pd.Series(final.loc[final.Age <=10, 'Fare']).values
age_20 = pd.Series(final.loc[(final.Age > 10) & (final.Age <= 20), 'Fare']).values
age_30 = pd.Series(final.loc[(final.Age > 20) & (final.Age <= 30), 'Fare']).values
age_40 = pd.Series(final.loc[(final.Age > 30) & (final.Age <= 40), 'Fare']).values

fig = plt.figure()
#plt.title('Age Vs Ticket Fare')
#plt.xlabel('Age')
#plt.ylabel('Ticket Fare')
#plt.ylim(0,300)
#plt.scatter(final['Age'],final['Fare'])
#plt.show()
#plt.figure()
#plt.xlim(0,320)
#bins = np.arange(0, 320, 20)
#plt.hist(final['Fare'], bins)
#plt.show()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.hist(age_10)
ax2.hist(age_20)
ax3.hist(age_30)
ax4.hist(age_40)
plt.show()
#final.to_csv('E:/Kaggle_Titanic/results_age_fare.csv', index=False)



