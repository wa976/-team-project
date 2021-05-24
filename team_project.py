# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:49:53 2021

@author: seung gyu
"""

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from collections import Counter

# load data
df = pd.read_csv('C:/Users/승규/Desktop/승규/3-1/머신러닝입문/team project/weatherAUS.csv')

df.info()
df.head()


#data visualization
fig, ax =plt.subplots(3,1)
plt.figure(figsize=(10,10))
sns.countplot(data=df,x='WindDir9am',ax=ax[0])
sns.countplot(data=df,x='WindDir3pm',ax=ax[1])
sns.countplot(data=df,x='WindGustDir',ax=ax[2])
fig.tight_layout()


fig, ax =plt.subplots(2,1)
plt.figure(figsize=(10,10))
sns.boxplot(df['Humidity3pm'],orient='v',color='c',ax=ax[0])
sns.boxplot(df['Humidity9am'],orient='v',color='c',ax=ax[1])
fig.tight_layout()


fig, ax =plt.subplots(2,1)
plt.figure(figsize=(10,10))
sns.boxplot(df['Pressure3pm'],orient='v',color='c',ax=ax[0])
sns.boxplot(df['Pressure9am'],orient='v',color='c',ax=ax[1])
fig.tight_layout()

sns.violinplot(x='RainToday',y='MaxTemp',data=df,hue='RainTomorrow')
plt.show()
sns.violinplot(x='RainToday',y='MinTemp',data=df,hue='RainTomorrow')
plt.show()

#범주형변수를 계량형 변수로 인코딩
class_le = LabelEncoder()
df['RainTomorrow'] = class_le.fit_transform(df['RainTomorrow'].values)
df['Location'] = class_le.fit_transform(df['Location'].values)
df['WindGustDir'] = class_le.fit_transform(df['WindGustDir'].values)
df['WindDir9am'] = class_le.fit_transform(df['WindDir9am'].values)
df['WindDir3pm'] = class_le.fit_transform(df['WindDir3pm'].values)
df['RainToday'] = class_le.fit_transform(df['RainToday'].values)

#각 변수별 연관성을 나타내는 표 
plt.figure(figsize=(15,15))
ax = sns.heatmap(df.corr(), square=True, annot=True, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)          
plt.show()




#null 값의 비율이 30% 이상인 colums 제거
df.isnull().mean()
df=df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis = 1)
df.isnull().mean()

# 강수량과 관련없는 columns 제거
df=df.drop(['Date'],axis = 1)
#df=df.drop(['Humidity3pm'],axis = 1)
#df=df.drop(['Temp3pm'],axis = 1)
#df=df.drop(['Temp9am'],axis = 1)

#null값을 삭제
df.isna().sum()
df = df.dropna()
df.info()
df.isna().sum()



#split data(x값과 y값 지정)
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

#test,train data로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2)

#LogisticRegression
LR = LogisticRegression(random_state=42)
LR.fit(X_train, y_train)
LR_score = LR.score(X_test, y_test)


#DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
DTC_score = tree.score(X_test, y_test)

#RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)
rfc.fit(X_train, y_train)
RFC_score = rfc.score(X_test, y_test)

print('linear Regression = ', LR_score,
      '\nDecisionTreeClassifier = ', DTC_score,
      '\nRandomForestClassifier = ', RFC_score)

