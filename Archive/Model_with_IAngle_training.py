# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:39:11 2017

@author: jlowe
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
from xgboost import XGBRegressor #sklearn wrapper for XGBoost
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random_seed=(9)

#Load data
os.chdir("C:/Users/jlowe/Documents/Projects/Kaggle - statoil iceberg prj/")
train = pd.read_json("train.json")
#test = pd.read_json("test.json")
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
#%%
'''Format the Data'''
# Train data
def data_wrangling(df, y):    
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    X = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle = np.array(df.inc_angle)
    if y == True:
        y = np.array(df["is_iceberg"])
        return X, X_angle, y
    else:
        return X, X_angle

X_Train, X_Angle_Train, y_Train = data_wrangling(train, y=True)
#%%
'''Predict Incident andgle using XGBoost'''
#split data to those with & without incident angle 
train_X = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1]*X_Train.shape[2]*X_Train.shape[3]))
train_predict_angle = np.concatenate((train_X, X_Angle_Train.reshape(X_Angle_Train.shape[0],1)), axis =1)
train_predict_angle = pd.DataFrame(train_predict_angle)

scaler = MinMaxScaler(feature_range=(0, 1))
train_predict_angle = scaler.fit_transform(train_predict_angle)
train_predict_angle = pd.DataFrame(train_predict_angle)
 
df_0 = train_predict_angle[train_predict_angle[16875] == 0] #seperate out the dataframe
df_t = train_predict_angle[train_predict_angle[16875] > 0] #seperate out the dataframe

df_0_x = df_0.iloc[:,:-1]
df_0_y = df_0.iloc[:,-1]
df_t_x = df_t.iloc[:,:-1]
df_t_y = df_t.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(df_t_x, df_t_y, test_size=0.1, random_state=42)
#%%   
model_BT = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100,
                                objective='reg:logistic',
                                gamma=0.2, min_child_weight=1, max_delta_step=0,
                                subsample=0.5, colsample_bytree=0.1, colsample_bylevel=1,
                                reg_alpha=0, reg_lambda=1, scale_pos_weight=0,seed=10)
model_BT.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

y_tr_pred = model_BT.predict(X_train)
y_te_pred  = model_BT.predict(X_test)

'''Train & Test error plotting'''
rmse_train = sqrt(mean_squared_error(y_train, y_tr_pred))
rmse_test = sqrt(mean_squared_error(y_test, y_te_pred))
# Function to calculate RMS errors for both train and test errors 
print('Test RMSE: %.3f' % rmse_test)
print('Train RMSE: %.3f' % rmse_train)
from util_recovery_plots import post_mortem
PM = post_mortem()
PM.plot_train_test(y_train, y_tr_pred,y_test, y_te_pred)

#tying to predict y, then add it onto the end of x. Invert scale. Then slot back into the original dataframe
#Make prediction of incident angle
df_0_x = pd.DataFrame(df_0_x)
df_0_y_p  = model_BT.predict(df_0_x)
# join the incident angle predictions with its associated image
df_0_x = np.array(df_0_x)
df_0_y_p = np.reshape(df_0_y_p,(df_0_y_p.shape[0],1))
df_0_p = np.concatenate((df_0_x,df_0_y_p), axis=1)
#perform the inverse scaling but first adding all the other columns
df_0_p = pd.DataFrame(df_0_p)
df_0_p_inv = scaler.inverse_transform(df_0_p)
#drop the colunms to leave the incident angle
df_0_p_inv = df_0_p_inv[:,-1]
#%%

# insert the valuse into X_train_angle to replace the 0's
#X_angle_train = np.reshape(X_angle_train,(X_angle_train.shape[0],1))
#df_0_p_inv = np.reshape(df_0_p_inv,(df_0_p_inv.shape[0],1))
X_angle_train = np.array(X_Angle_Train)
df_0_p_inv2 = np.array(df_0_p_inv)

insert = 0
for i in range(0, len(X_angle_train)):
    if X_angle_train[i,] == 0:
        X_angle_train[i,] = df_0_p_inv2[insert,]
        insert += 1
       
X_Angle_Train = X_angle_train         
        
#%%
'''Feature normalisation'''


#%%
'''Sliding window to extract iceberg or boat'''


#%%
'''feature augmentation'''
#Curl, gradient etc...


#%%
'''Convolutional Neural Network'''


#%% 
'''Image Plotting, f1 score & confusion matrix'''


#%%
'''Test data predict and csv export'''