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
# load data
df = pd.read_csv('C:/Users/승규/Desktop/승규/3-1/머신러닝입문/team project/weatherAUS.csv')

df.info()
df.head()


# 변수의 성질에 따라 df 나누기

float_data = list(filter(lambda x: df[x].dtypes !='object', df.columns))
object_data = list(filter(lambda x: df[x].dtypes !='float64', df.columns))



# null 값을 평균값으로 채우기
def get_not_nan_value(table):
    return table.fillna(value=table.mean())

def get_kde_x_not_nan_value(table: pd.Series):
    """Return KDE and X: massive."""
    without_nan = get_not_nan_value(table)
    kde = gaussian_kde(without_nan)
    x = np.linspace(without_nan.min(), without_nan.max(), 100)
    return kde, x, without_nan

def draw_hist_and_density(kde, x, value,
                               title_text='', legend_text=None,
                               x_text='', y_text=''):
    """This function draw hist and density."""
    plt.plot(x, kde(x), color='g')
    plt.hist(value, density=True)
    plt.title(title_text)
    plt.legend(legend_text) if legend_text is not None else None
    plt.xlabel(x_text)
    plt.ylabel(y_text)


def draw_hist(value, title_text='',
              x_text='', y_text=''):
    """This function draw hist."""
    plt.hist(value, density=True)
    plt.title(title_text)
    plt.xlabel(x_text)
    plt.ylabel(y_text)

    
def draw_scatter_plot(x_, y_, alpha_, title_text='', 
                      x_text='', y_text=''):
    """This function draw dependencies."""
    plt.scatter(x=x_, y=y_, alpha=alpha_)
    plt.title(title_text)
    plt.xlabel(x_text)
    plt.ylabel(y_text)

plt.figure(figsize=(12, 7))



kde_min, x_min, min_temp = get_kde_x_not_nan_value(df['MinTemp'])
kde_max, x_max, max_temp = get_kde_x_not_nan_value(df['MaxTemp'])
kde_rain, x_rain, rain = get_kde_x_not_nan_value(df['Rainfall'])
kde_evp, x_evp, evaporation = get_kde_x_not_nan_value(df['Evaporation'])

plt.subplot(2, 2, 1)
draw_hist_and_density(kde=kde_min, x=x_min, value=min_temp,
                           title_text='MinTemp',
                           legend_text=[
                               'distribution density',
                               '$t_{min}$ distribution'
                           ])
plt.subplot(2, 2, 2)
draw_hist_and_density(kde=kde_max, x=x_max, value=max_temp,
                           title_text='MaxTemp',
                           legend_text=[
                               'distribution density',
                               '$t_{max}$ distribution'
                           ])

plt.subplot(2, 2, 3)
draw_hist_and_density(kde=kde_rain, x=x_rain, value=rain,
                      title_text='Rainfall',
                      legend_text=[
                          'distribution density',
                          'Rainfall distribution'
                      ])

plt.subplot(2, 2, 4)
draw_hist_and_density(kde=kde_evp, x=x_evp, value=evaporation,
                           title_text='Evaporation',
                           legend_text=[
                               'distribution density',
                               'Evaporation distribution'
                           ])


plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
draw_scatter_plot(x_=min_temp, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the MinTemp", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,2)
draw_scatter_plot(x_=max_temp, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the MaxTemp", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,3)
draw_scatter_plot(x_=rain, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Rainfal", 
                  x_text='Rainfal', y_text='Rainfall')
plt.subplot(2,2,4)
draw_scatter_plot(x_=evaporation, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Evaporation", 
                  x_text='Evaporation', y_text='Rainfall')


plt.figure(figsize=(12, 7))
kde_sunshine, x_sunshine, sunshine = get_kde_x_not_nan_value(df['Sunshine'])
kde_wind, x_wind, wind_gust_speed = get_kde_x_not_nan_value(df['WindGustSpeed'])
kde_winds_am, x_winds_am, wind_speed_am = get_kde_x_not_nan_value(df['WindSpeed9am'])
kde_winds_pm, x_winds_pm, wind_speed_pm = get_kde_x_not_nan_value(df['WindSpeed3pm'])

plt.subplot(2, 2, 1)
draw_hist_and_density(kde=kde_sunshine, x=x_sunshine, value=sunshine,
                      title_text='Sunshine',
                      legend_text=[
                          'distribution density',
                          'Sunshine distribution'
                      ])

plt.subplot(2, 2, 2)
draw_hist_and_density(kde=kde_wind, x=x_wind, value=wind_gust_speed,
                           title_text='WindGustSpeed',
                           legend_text=[
                               'distribution density',
                               'WindGustSpeed distribution'
                           ])

plt.subplot(2, 2, 3)
draw_hist_and_density(kde=kde_winds_am, x=x_winds_am, value=wind_speed_am,
                           title_text='WindSpeed9am',
                           legend_text=[
                               'distribution density',
                               'WindSpeed9am distribution'
                           ])

plt.subplot(2, 2, 4)
draw_hist_and_density(kde=kde_winds_pm, x=x_winds_pm, value=wind_speed_pm,
                           title_text='WindSpeed3pm',
                           legend_text=[
                               'distribution density',
                               'WindSpeed3pm distribution'
                           ])


plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
draw_scatter_plot(x_=sunshine, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Sunshine", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,2)
draw_scatter_plot(x_=wind_gust_speed, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the WindGustSpeed", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,3)
draw_scatter_plot(x_=wind_speed_am, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the WindSpeed9am", 
                  x_text='WindSpeed9am', y_text='Rainfall')
plt.subplot(2,2,4)
draw_scatter_plot(x_=wind_speed_pm, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the WindSpeed3pm", 
                  x_text='WindSpeed3pm', y_text='Rainfall')

plt.figure(figsize=(12, 7))
kde_hum_am, x_hum_am, humidity_am = get_kde_x_not_nan_value(df['Humidity9am'])
kde_hum_pm, x_hum_pm, humidity_pm = get_kde_x_not_nan_value(df['Humidity3pm'])
kde_pres_am, x_pres_am, pressure_am = get_kde_x_not_nan_value(df['Pressure9am'])
kde_pres_pm, x_pres_pm, pressure_pm = get_kde_x_not_nan_value(df['Pressure3pm'])

plt.subplot(2, 2, 1)
draw_hist_and_density(kde=kde_hum_am, x=x_hum_am, value=humidity_am,
                           title_text='Humidity9am',
                           legend_text=[
                               'distribution density',
                               'Humidity9am distribution'
                           ])

plt.subplot(2, 2, 2)
draw_hist_and_density(kde=kde_hum_pm, x=x_hum_pm, value=humidity_pm,
                           title_text='Humidity3pm',
                           legend_text=[
                               'Humidity3pm density',
                               'WindSpeed3pm distribution'
                           ])

plt.subplot(2, 2, 3)
draw_hist_and_density(kde=kde_pres_am, x=x_pres_am, value=pressure_am,
                           title_text='Pressure9am',
                           legend_text=[
                               'distribution density',
                               'Pressure9am distribution'
                           ])

plt.subplot(2, 2, 4)
draw_hist_and_density(kde=kde_pres_pm, x=x_pres_pm, value=pressure_pm,
                           title_text='Pressure3pm',
                           legend_text=[
                               'Pressure3pm density',
                               'Pressure3pm distribution'
                           ])


plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
draw_scatter_plot(x_=humidity_am, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Humidity9am", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,2)
draw_scatter_plot(x_=humidity_pm, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Humidity3pm", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,3)
draw_scatter_plot(x_=pressure_am, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Pressure9am", 
                  x_text='Pressure9am', y_text='Rainfall')
plt.subplot(2,2,4)
draw_scatter_plot(x_=pressure_pm, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Pressure3pm", 
                  x_text='Pressure3pm', y_text='Rainfall')


plt.figure(figsize=(12, 7))
kde_cloud_am, x_cloud_am, cloud_am = get_kde_x_not_nan_value(df['Cloud9am'])
kde_cloud_pm, x_cloud_pm, cloud_pm = get_kde_x_not_nan_value(df['Cloud3pm'])
kde_temp_am, x_temp_am, temp_am = get_kde_x_not_nan_value(df['Temp9am'])
kde_temp_pm, x_temp_pm, temp_pm = get_kde_x_not_nan_value(df['Temp3pm'])

plt.subplot(2, 2, 1)
draw_hist_and_density(kde=kde_cloud_am, x=x_cloud_am, value=cloud_am,
                           title_text='Cloud9am',
                           legend_text=[
                               'distribution density',
                               'Cloud9am distribution'
                           ])

plt.subplot(2, 2, 2)
draw_hist_and_density(kde=kde_cloud_pm, x=x_cloud_pm, value=cloud_pm,
                           title_text='Cloud3pm',
                           legend_text=[
                               'distribution density',
                               'Cloud3pm distribution'
                           ])

plt.subplot(2, 2, 3)
draw_hist_and_density(kde=kde_temp_am, x=x_temp_am, value=temp_am,
                           title_text='Temp9am',
                           legend_text=[
                               'distribution density',
                               'Temp9am distribution'
                           ])

plt.subplot(2, 2, 4)
draw_hist_and_density(kde=kde_temp_pm, x=x_temp_pm, value=temp_pm,
                           title_text='Temp3pm',
                           legend_text=[
                               'distribution density',
                               'Temp3pm distribution'
                           ])


plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
draw_scatter_plot(x_=cloud_am, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Cloud9am", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,2)
draw_scatter_plot(x_=cloud_pm, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Cloud3pm", 
                  x_text='', y_text='Rainfall')
plt.subplot(2,2,3)
draw_scatter_plot(x_=temp_am, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Temp9am", 
                  x_text='Temp9am', y_text='Rainfall')
plt.subplot(2,2,4)
draw_scatter_plot(x_=temp_pm, y_=rain, alpha_=0.02, 
                  title_text="Rain's dependence on the Temp3pm", 
                  x_text='Temp3pm', y_text='Rainfall')


# 강수량과 관련있는 column만 추출
cols_need = [
    'Location', 'MinTemp', 'MaxTemp',
    'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
    'WindSpeed3pm', 'Pressure9am', 'Pressure3pm',
    'Temp9am', 'Temp3pm', 'RainTomorrow'
]

new_df = df[cols_need]

new_df.info()
new_df.isna().sum()

#null값을 삭제
new_df = new_df.dropna()
new_df.info()
new_df.isna().sum()

#범주형변수를 계량형 변수로 인코딩
class_le = LabelEncoder()
new_df['RainTomorrow'] = class_le.fit_transform(new_df['RainTomorrow'].values)
new_df['Location'] = class_le.fit_transform(new_df['Location'].values)
new_df.info()

#split data(x값과 y값 지정)
X = new_df.iloc[:, :-1]
y = new_df.iloc[:, -1:]

#test,train data로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.3)

#LogisticRegression
LR = LogisticRegression(random_state=42)
LR.fit(X_train, y_train)

lr_head = LR.predict(X_test)
print(f"""
accuracy_score: {accuracy_score(lr_head, y_test)}
roc_auc_score: {roc_auc_score(lr_head, y_test)}
""")

#DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)


#RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
