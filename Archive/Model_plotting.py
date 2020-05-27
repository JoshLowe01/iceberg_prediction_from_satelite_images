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
import cv2
np.random_seed=(9)

#Load data
os.chdir("C:/Users/jlowe/Documents/Projects/Kaggle - statoil iceberg prj/")
train = pd.read_json("train.json")
#test = pd.read_json("test.json")
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
#%%
'''Denoising'''
def Denoise(df, filter_strength):
    #seperate out the two images
    bands_1 = np.stack([np.array(band).reshape(75, 75) for band in df['band_1']], axis=0)
    bands_2 = np.stack([np.array(band).reshape(75, 75) for band in df['band_2']], axis=0)
    #in order to apply the filter the data needs to be scaled like a regular image channel ie. 0 to 255
    scaler_band1 = MinMaxScaler(feature_range=(0, 255))
    scaler_band2 = MinMaxScaler(feature_range=(0, 255))
    #Apply the transforms and reshape to original shape
    bands_1 = scaler_band1.fit_transform(bands_1.reshape((bands_1.shape[0],bands_1.shape[1]*bands_1.shape[2])))
    bands_2 = scaler_band2.fit_transform(bands_2.reshape((bands_2.shape[0],bands_2.shape[1]*bands_2.shape[2])))
    bands_1 = bands_1.reshape((bands_1.shape[0],75,75))
    bands_2 = bands_2.reshape((bands_2.shape[0],75,75))
    tmp1 = list()
    tmp2 = list()
        #apply the filter to each image, unfortunelty i have to scale to uint8 -> am i loosing resolution?
    for i in range(0, df.shape[0]):
        tmp11 = cv2.fastNlMeansDenoising(np.uint8(bands_1[i].reshape((75,75))),None,filter_strength,7,21)
        tmp11 = tmp11.reshape((tmp11.shape[0]*tmp11.shape[1]))
        tmp1.append(tmp11)
        tmp22 = cv2.fastNlMeansDenoising(np.uint8(bands_2[i].reshape((75,75))),None,filter_strength,7,21)
        tmp22 = tmp22.reshape((tmp22.shape[0]*tmp22.shape[1]))
        tmp2.append(tmp22)
    #replace the columns within the original df
    df['band_1'] = tmp1
    df['band_2'] = tmp2
    return df

#%%
#'''scaling Band 1 and Band 2'''
#def B1_B2_scaling(df):
#        bands_1 = np.stack([np.array(band).reshape(75, 75) for band in df['band_1']], axis=0)
#        bands_2 = np.stack([np.array(band).reshape(75, 75) for band in df['band_2']], axis=0)
#        #scale between 0 & 1
#        scaler_band1 = MinMaxScaler(feature_range=(0, 1))
#        scaler_band2 = MinMaxScaler(feature_range=(0, 1))
#        #fit transform to flatteneed image 
#        bands_1 = scaler_band1.fit_transform(bands_1.reshape((bands_1.shape[0],bands_1.shape[1]*bands_1.shape[2])))
#        bands_2 = scaler_band2.fit_transform(bands_2.reshape((bands_2.shape[0],bands_2.shape[1]*bands_2.shape[2])))
#        bands_1 = bands_1.reshape((bands_1.shape[0],75,75))
#        bands_2 = bands_2.reshape((bands_2.shape[0],75,75))
#%%
'''Predict Incident andgle using XGBoost'''
def predict_IAngle(df):   
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])  
    X_Angle_Train = np.array(df.inc_angle)
    X_Train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], ((x_band1/(x_band2+0.000000001)))[:, :, :, np.newaxis]], axis=-1) #also adding a term to make it a 3 channel image
    #split data to those with & without incident angle 
    train_X = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1]*X_Train.shape[2]*X_Train.shape[3]))
    train_predict_angle = np.concatenate((train_X, X_Angle_Train.reshape(X_Angle_Train.shape[0],1)), axis =1)
    train_predict_angle = pd.DataFrame(train_predict_angle)
    #Scale the features for the IAngle prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_predict_angle = np.array(train_predict_angle, dtype='float64')
    train_predict_angle = scaler.fit_transform(train_predict_angle)
    train_predict_angle = pd.DataFrame(train_predict_angle)
     
    df_0 = train_predict_angle[train_predict_angle[16875] == 0] #seperate out the dataframe
    df_t = train_predict_angle[train_predict_angle[16875] > 0] #seperate out the dataframe
    #Split X and Y
    df_0_x = df_0.iloc[:,:-1]
#    df_0_y = df_0.iloc[:,-1]
    df_t_x = df_t.iloc[:,:-1]
    df_t_y = df_t.iloc[:,-1]
    #Define the predictive model
    model_BT = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=100,
                                    objective='reg:logistic',
                                    gamma=0.2, min_child_weight=1, max_delta_step=0,
                                    subsample=0.5, colsample_bytree=0.1, colsample_bylevel=1,
                                    reg_alpha=0, reg_lambda=1, scale_pos_weight=0,seed=10)
    model_BT.fit(df_t_x, df_t_y, verbose=True)
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
    #reinsert the predicted IAngle
    insert = 0
    for i in range(0, len(X_Angle_Train)):
        if X_Angle_Train[i,] == 0:
            X_Angle_Train[i,] = df_0_p_inv[insert,]
            insert += 1
    df['inc_angle'] = X_Angle_Train
    return df
#%%
'''Apply scaling to Bands 1 and 2 to counteract different incident angle aquisitions.
    Similar to zero offseting in seismic reflection data'''
def Incident_Angle_scaling(df, scaler):
    #format bands 1 and 2 ready for the caler to be applied
    bands_1 = np.stack([np.array(band).reshape(75, 75) for band in df['band_1']], axis=0)
    bands_2 = np.stack([np.array(band).reshape(75, 75) for band in df['band_2']], axis=0)
    #scale between 0 & 1
    scaler_band1 = MinMaxScaler(feature_range=(0, 1))
    scaler_band2 = MinMaxScaler(feature_range=(0, 1))
    #fit transform to flatteneed image 
    bands_1 = scaler_band1.fit_transform(bands_1.reshape((bands_1.shape[0],bands_1.shape[1]*bands_1.shape[2])))
    bands_2 = scaler_band2.fit_transform(bands_2.reshape((bands_2.shape[0],bands_2.shape[1]*bands_2.shape[2])))
    #centre the incident angle around 38 and adjust accordingly
    IA_scaled = np.array(df['inc_angle'])
    IA_scaled = 38. - IA_scaled #calculate how much the angle deviates from the "central" angle of 38
    IA_scaler = IA_scaled*scaler # multiply the deviation by the scaler to get the amount to add onto bands 1 and 2
    #apply the scaling to each band
    bands_1 = bands_1 - IA_scaler.reshape(IA_scaler.shape[0],1)  
    bands_2 = bands_2 - IA_scaler.reshape(IA_scaler.shape[0],1) 
    #Inverse scaling
    bands_1 = scaler_band1.inverse_transform(bands_1) 
    bands_2 = scaler_band2.inverse_transform(bands_2) 
    #reasemble the dataframe, must be a better way than this but it does the job
    tmp1 = list()
    tmp2 = list()
    for row in range(0,bands_1.shape[0]):
         tmp1.append(list(bands_1[row,:]))
         tmp2.append(list(bands_2[row,:]))
    df['band_1'] = tmp1
    df['band_2'] = tmp2
    return df
#%%
'''feature augmentation'''
def dx_dy(df, ksize):
    #seperate out the two images
    bands_1 = np.stack([np.array(band).reshape(75, 75) for band in df['band_1']], axis=0)
    bands_2 = np.stack([np.array(band).reshape(75, 75) for band in df['band_2']], axis=0)
    #in order to apply the filter the data needs to be scaled like a regular image channel ie. 0 to 255
    scaler_band1 = MinMaxScaler(feature_range=(0, 255))
    scaler_band2 = MinMaxScaler(feature_range=(0, 255))
    #Apply the transforms and reshape to original shape
    bands_1 = scaler_band1.fit_transform(bands_1.reshape((bands_1.shape[0],bands_1.shape[1]*bands_1.shape[2])))
    bands_2 = scaler_band2.fit_transform(bands_2.reshape((bands_2.shape[0],bands_2.shape[1]*bands_2.shape[2])))
    bands_1 = bands_1.reshape((bands_1.shape[0],75,75))
    bands_2 = bands_2.reshape((bands_2.shape[0],75,75))
    tmp1 = list()
    tmp2 = list()
    tmp3 = list()
    tmp4 = list()
        #apply the filter to each image, unfortunelty i have to scale to uint8 -> am i loosing resolution?
    for i in range(0, df.shape[0]):
        #X derivative for both channels
        tmp11 = cv2.Sobel(bands_1[i].reshape((75,75)),cv2.CV_64F,1,0,ksize=ksize)
        tmp11 = tmp11.reshape((tmp11.shape[0]*tmp11.shape[1]))
        tmp1.append(tmp11)
        tmp22 = cv2.Sobel(bands_2[i].reshape((75,75)),cv2.CV_64F,1,0,ksize=ksize)
        tmp22 = tmp22.reshape((tmp22.shape[0]*tmp22.shape[1]))
        tmp2.append(tmp22)
        #Y derivative for both channels
        tmp31 = cv2.Sobel(bands_1[i].reshape((75,75)),cv2.CV_64F,0,1,ksize=ksize)
        tmp31 = tmp11.reshape((tmp31.shape[0]*tmp31.shape[1]))
        tmp3.append(tmp31)
        tmp42 = cv2.Sobel(bands_2[i].reshape((75,75)),cv2.CV_64F,0,1,ksize=ksize)
        tmp42 = tmp22.reshape((tmp42.shape[0]*tmp42.shape[1]))
        tmp4.append(tmp42)
    #replace the columns within the original df
    df['band_1_dx'] = tmp1
    df['band_2_dx'] = tmp2
    df['band_1_dy'] = tmp3
    df['band_2_dy'] = tmp4
    return df
#laplacian = cv2.Laplacian(img,cv2.CV_64F)
#Curl, gradient etc...
#%%
'''Format the Data'''
# Train data
def data_wrangling(df, y):    
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    x_band3 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1_dx"]])
    x_band4 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2_dx"]])
    x_band5 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1_dy"]])
    x_band6 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2_dy"]])
    X = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , ((x_band1/(x_band2+0.000000001)))[:, :, :, np.newaxis]], axis=-1) #also adding a term to make it a 3 channel image
    X_super = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                              , x_band3[:, :, :, np.newaxis]
                              , x_band4[:, :, :, np.newaxis]
                              , x_band5[:, :, :, np.newaxis]
                              , x_band6[:, :, :, np.newaxis]
                              , ((x_band1*x_band2))[:, :, :, np.newaxis]], axis=-1) #also adding a term to make it a 3 channel image
    X_angle = np.array(df.inc_angle)
    if y == True:
        y = np.array(df["is_iceberg"])
        return X, X_super, X_angle, y
    else:
        return X, X_super, X_angle


#%%


#%%
'''Feature normalisation'''
def Scaling(X):
    X = np.uint8(X)
    X = cv2.normalize(X, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return X

#%%
'''Pipeline'''
#Denoising
train = Denoise(train, filter_strength=10)
#Adding features
train = dx_dy(train, ksize=5)
#Calculate missing Incident Angles
train = predict_IAngle(train)
#Apply scaler based on Incident Angle of aquisition.
train = Incident_Angle_scaling(train, scaler=0.015)
#Data wrangling
X_Train, X_Train_extreme, X_Angle_Train, y_Train = data_wrangling(train, y=True)
#scaling
X_Train = Scaling(X_Train)
X_Train_extreme = Scaling(X_Train_extreme)
  
#%%
'''Feature normalisation'''
#Multiplying each pixel value in the image by cos(incident angle), the aim is to bring each image to zero offset
#X_Train = np.transpose(X_Train) * np.cos(X_Angle_Train)
#X_Train = np.transpose(X_Train)
tmp_all = list()
for i in range(0, len(X_Train)):
    tmp1 = np.median((X_Train[i,:,:,0]))
    tmp2 = np.median((X_Train[i,:,:,1]))
    tmp3 = np.median((X_Train[i,:,:,2]))
    tmp4 = X_Angle_Train[i]
    tmp_all.append((tmp1, tmp2, tmp3, tmp4))
tmp_all = np.vstack(tmp_all)

from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
plt.plot(tmp_all[:,3],tmp_all[:,0], 'rs')
plt.plot(tmp_all[:,3],tmp_all[:,1], 'gs')
plt.plot(tmp_all[:,3],tmp_all[:,2]*20, 'bs')
plt.plot(np.unique(tmp_all[:,0]), np.poly1d(np.polyfit(tmp_all[:,0], tmp_all[:,3], 1))(np.unique(tmp_all[:,0])))
plt.plot(np.unique(tmp_all[:,1]), np.poly1d(np.polyfit(tmp_all[:,1], tmp_all[:,3], 1))(np.unique(tmp_all[:,1])))
plt.plot(np.unique(tmp_all[:,2]), np.poly1d(np.polyfit(tmp_all[:,2], tmp_all[:,3], 1))(np.unique(tmp_all[:,2])))
plt.ylim()#0, 40)
plt.xlim(25, 50)
plt.show()

#%%
'''Convolutional Neural Network'''
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense
def CNN():
    model = Sequential()
    model.add(BatchNormalization(input_shape = (75, 75, 3)))
    for i in range(4):
        model.add(Conv2D(8*2**i, kernel_size = (3,3)))
        model.add(MaxPooling2D((2,2)))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

model = CNN()
model.fit(X_Train, y_Train, validation_split=0.2, batch_size=10, epochs = 10, verbose=1)
#%% 
'''Image Plotting, f1 score & confusion matrix'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from pylab import rcParams

plt.rcParams['figure.figsize'] = 10, 10

df = train
bands_1 = np.stack([np.array(band).reshape(75, 75) for band in df['band_1']], axis=0)
bands_2 = np.stack([np.array(band).reshape(75, 75) for band in df['band_2']], axis=0)

v_max = max(abs(bands_1).max(), abs(bands_2).max())
v_min = -v_max

samples_ship = df.index[df['is_iceberg'] == 0]
samples_iceberg = df.index[df['is_iceberg'] == 1]
#%%
dst = bands_1[10].reshape((75,75))
scaler_band1 = MinMaxScaler(feature_range=(0, 255))
scaler_band2 = MinMaxScaler(feature_range=(0, 255))
bands_1 = scaler_band1.fit_transform(bands_1.reshape((bands_1.shape[0],bands_1.shape[1]*bands_1.shape[2])))
bands_2 = scaler_band2.fit_transform(bands_2.reshape((bands_2.shape[0],bands_2.shape[1]*bands_2.shape[2])))
bands_1 = bands_1.reshape((bands_1.shape[0],75,75))
bands_2 = bands_2.reshape((bands_2.shape[0],75,75))
v_max = max(abs(bands_1).max(), abs(bands_2).max())
v_min = -v_max
#%%
rcParams['figure.figsize'] = 30, 30
N_plots = 10
fig, axss = plt.subplots(N_plots, 8)
for i_plot, axs in enumerate(axss):
    sample_ship = samples_ship[i_plot]
    sample_iceberg = samples_iceberg[i_plot]
    dst1 = cv2.fastNlMeansDenoising(np.uint8(bands_1[sample_ship].reshape((75,75))),None,10,7,21)
    dst2 = np.uint8(bands_1[sample_ship])
    dst3 = cv2.fastNlMeansDenoising(np.uint8(bands_2[sample_ship].reshape((75,75))),None,10,7,21)
    dst4 = np.uint8(bands_2[sample_ship])
    dst5 = cv2.fastNlMeansDenoising(np.uint8(bands_1[sample_iceberg].reshape((75,75))),None,10,7,21) 
    dst6 = np.uint8(bands_1[sample_iceberg])
    dst7 = cv2.fastNlMeansDenoising(np.uint8(bands_2[sample_iceberg].reshape((75,75))),None,10,7,21) 
    dst8 = np.uint8(bands_2[sample_iceberg])
    axs[0].imshow(dst1, vmin=v_min, vmax=v_max)
    axs[1].imshow(dst2, vmin=v_min, vmax=v_max)
    axs[2].imshow(dst3, vmin=v_min, vmax=v_max)
    axs[3].imshow(dst4, vmin=v_min, vmax=v_max)
    axs[4].imshow(dst5, vmin=v_min, vmax=v_max)
    axs[5].imshow(dst6, vmin=v_min, vmax=v_max)
    axs[6].imshow(dst7, vmin=v_min, vmax=v_max)
    axs[7].imshow(dst8, vmin=v_min, vmax=v_max)


#%%
rcParams['figure.figsize'] = 10, 10
N_plots = 3
fig, axss = plt.subplots(N_plots, 7)
for i_plot, axs in enumerate(axss):
    sample_ship = samples_ship[i_plot]
    sample_iceberg = samples_iceberg[i_plot]
    dst1 = cv2.Sobel(bands_1[sample_ship].reshape((75,75)),cv2.CV_64F,1,0,ksize=5)
    dst11 = cv2.Sobel(bands_1[sample_ship].reshape((75,75)),cv2.CV_64F,0,1,ksize=5)
    dst12 = cv2.fastNlMeansDenoising(np.uint8(bands_1[sample_ship].reshape((75,75))),None,10,7,21)
    dst121 = cv2.Sobel(dst12,cv2.CV_64F,1,0,ksize=5)
    dst122 = cv2.Sobel(dst12,cv2.CV_64F,0,1,ksize=5)
    dst1211 = cv2.fastNlMeansDenoising(np.uint8(dst1),None,2,7,7)
    dst1222 = cv2.fastNlMeansDenoising(np.uint8(dst11),None,2,7,7)
    dst2 = np.uint8(bands_1[sample_ship])
    dst3 = cv2.Laplacian(bands_2[sample_ship].reshape((75,75)),ddepth=-50)
    dst4 = np.uint8(bands_2[sample_ship])
    dst5 = cv2.Laplacian(bands_1[sample_iceberg].reshape((75,75)),ddepth=-1)
    dst6 = np.uint8(bands_1[sample_iceberg])
    dst7 = cv2.Laplacian(bands_1[sample_iceberg].reshape((75,75)),ddepth=-1)
    dst8 = np.uint8(bands_2[sample_iceberg])
    axs[0].imshow(dst1)#, vmin=v_min, vmax=v_max)
    axs[1].imshow(dst11)#, vmin=v_min, vmax=v_max)
    axs[2].imshow(dst2)#, vmin=v_min, vmax=v_max)
    axs[3].imshow(dst121)#, vmin=v_min, vmax=v_max)
    axs[4].imshow(dst122)#, vmin=v_min, vmax=v_max)
    axs[5].imshow(dst1211)#, vmin=v_min, vmax=v_max)
    axs[6].imshow(dst1222)#, vmin=v_min, vmax=v_max)
#    axs[2].imshow(dst3, vmin=v_min, vmax=v_max)
#    axs[3].imshow(dst4, vmin=v_min, vmax=v_max)
#    axs[4].imshow(dst5, vmin=v_min, vmax=v_max)
#    axs[5].imshow(dst6, vmin=v_min, vmax=v_max)
#    axs[6].imshow(dst7, vmin=v_min, vmax=v_max)
#    axs[7].imshow(dst8, vmin=v_min, vmax=v_max)
    
#%%
alpha = 0.5
fig, axss = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
_ = axss[0,0].hist(bands_1[samples_ship, :, :].flatten(), 
                   bins=100, normed=True, alpha=alpha, label='ship')
_ = axss[0,1].hist(bands_2[samples_ship, :, :].flatten(), 
                   bins=100, normed=True, alpha=alpha, label='ship')
_ = axss[0,0].hist(bands_1[samples_iceberg, :, :].flatten(), 
                   bins=100, normed=True, alpha=alpha, label='iceberg')
_ = axss[0,1].hist(bands_2[samples_iceberg, :, :].flatten(), 
                   bins=100, normed=True, alpha=alpha, label='iceberg')
axss[0, 0].set_title('band_1')
axss[0, 1].set_title('band_2')
axss[0, 1].legend()
#%%
'''Test data predict and csv export'''