# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:49:53 2021

@author: seung gyu
"""

"load data"
import numpy as np
import pandas as pd
import seaborn as sns 

df = pd.read_csv('C:/Users/승규/Desktop/승규/3-1/머신러닝입문/team project/weatherAUS.csv')

df.head()

"각 cloumn 마다 null값의 비율 계산"
df.isnull().mean()


"null의 비율이 20%가 넘는 column삭제"

df=df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis = 1)

"범주형 변수와 계량형 변수를 분리"

df_cat=df[['WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow','Date','Location']]

df_num=df.drop(['WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow','Date','Location'], axis = 1)
