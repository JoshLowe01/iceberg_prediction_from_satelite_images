# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:39:11 2017

@author: jlowe
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor #sklearn wrapper for XGBoost
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import L1L2
from keras import optimizers
import time

np.random_seed=(9)
#%%
'''Import the Data'''
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
    tmp1, tmp2 = list(), list()
        #apply the filter to each image, unfortunelty i have to scale to uint8 -> am i loosing resolution?
    for i in range(0, df.shape[0]):
        tmp11 = cv2.fastNlMeansDenoising(np.uint8(bands_1[i].reshape((75,75))),None,filter_strength,7,21)
        tmp11 = tmp11.reshape((tmp11.shape[0]*tmp11.shape[1]))
        tmp1.append(tmp11)
        tmp22 = cv2.fastNlMeansDenoising(np.uint8(bands_2[i].reshape((75,75))),None,filter_strength,7,21)
        tmp22 = tmp22.reshape((tmp22.shape[0]*tmp22.shape[1]))
        tmp2.append(tmp22)
    #replace the columns within the original df
    df['band_1'], df['band_2'] = tmp1, tmp2
    return df

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
    #X_Super is an 8 channel image which can be split into two 4 channel images
    X_super = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                              , ((x_band1/(x_band2+0.00000001)))[:, :, :, np.newaxis]
                              , ((x_band1*x_band2))[:, :, :, np.newaxis]
                              , x_band3[:, :, :, np.newaxis]
                              , x_band4[:, :, :, np.newaxis]
                              , x_band5[:, :, :, np.newaxis]
                              , x_band6[:, :, :, np.newaxis]]
                              , axis=-1) #also adding a term to make it a 3 channel image
    #Split the X_Train_extreme into two 4 channel images
    X_super_regular = X_super[:,:,:,0:3]
    X_super_derivatives = X_super[:,:,:,4:7]
    
    X_angle = np.array(df.inc_angle)
    if y == True:
        y = np.array(df["is_iceberg"])
        return X, X_super_regular, X_super_derivatives, X_angle, y
    else:
        return X, X_super_regular, X_super_derivatives, X_angle

#%%
'''Feature normalisation'''
def Scaling(X):
    #scale all image channels between 0 and 1
    X = cv2.normalize(X, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return X

#%%
'''Image Augmentation'''
  
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
X_Train, X_Train_extreme_regular, X_Train_extreme_derivatives, X_Angle_Train, y_Train = data_wrangling(train, y=True)
#scaling
X_Train = Scaling(X_Train)
X_Train_extreme_regular = Scaling(X_Train_extreme_regular)
X_Train_extreme_derivatives = Scaling(X_Train_extreme_derivatives)

#Split the X_Train_extreme into two 4 channel images
#%%
'''Image Augmentation'''
#TODO need to find a way to do this with Multi Channel images
#TODO possibly need to gen the images then turn back to np arrays --> then run through pipeline to generate the other features
# Define the image transformations here
generator = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 40,
                         seed=7)

def generate_data_generator_for_two_images(X1, X2, Y):
    genX1 = generator.flow(X1,Y, batch_size=64, seed=7, shuffle=False)
    genX2 = generator.flow(X2, batch_size=64, seed=7, shuffle=False)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i ], X1i[1]
#    return gen_flow
#%%
'''Convolutional Neural Network'''
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense, Merge

#X_train = X_Train #choose X_Train_extreme to use all the added features
batch_size = 2

X_train1 = X_Train_extreme_regular
X_train2 = X_Train_extreme_derivatives

input_shape1 = (X_train1.shape[1],X_train1.shape[2],X_train1.shape[3])
input_shape2 = (X_train2.shape[1],X_train2.shape[2],X_train2.shape[3])

#Parameter
epochs = [10]#0, 200, 400, 800]#200
batchs = 5
learn_rate = [0.1,0.01,0.001]#, 0.0006, 0.0001, 0.00006, 0.00001, 0.000006, 0.000001, 0.0000006, 0.0000001]
neurons = [0]#, 128, 256, 512, 768, 1024, 2046]
L2REG = [0]#, 0.0006, 0.0001, 0.00006, 0.00001, 0.000006, 0.000001, 0.0000006, 0.0000001]
decay = [0]
Drop = [0]
Results=[]
count = 0

tmp=list() #empty list to append the values too

for neuron in neurons:
    for learnrate in learn_rate:
        for drop in Drop:
            for l1l2 in L1L2REG:
                for ep in epochs:
                    for dekay in decay:               
                        start = time.time()
                        #convert parameters to intergers
                        neuron = int(neuron)
                        learnrate = float(learnrate)
                        l1l2 = float(l1l2)
                        ep = int(ep)
                        dekay = float(dekay)
                        drop = float(drop)
                        
                        #Define parameters
                        regulisation = L1L2(l1=0, l2=l1l2) # take value from L1L2 list
                        Adam = optimizers.Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=dekay)

                        def CNN():
                            #two branches which take in the two 4 channel images and
                            branch1 = Sequential()
                            branch1.add(BatchNormalization(input_shape = input_shape1))
                            for i in range(4):
                                branch1.add(Conv2D(8*2**i, kernel_size = (3,3)))
                                branch1.add(MaxPooling2D((2,2)))
                            branch1.add(GlobalMaxPooling2D())
                            
                            branch2 = Sequential()
                            branch2.add(BatchNormalization(input_shape = input_shape2))
                            for i in range(4):
                                branch2.add(Conv2D(8*2**i, kernel_size = (3,3)))
                                branch2.add(MaxPooling2D((2,2)))
                            branch2.add(GlobalMaxPooling2D())
                            
                            model = Sequential()
                            model.add(Merge([branch1, branch2], mode = 'concat'))
                            model.add(Dropout(0.2))
                            model.add(Dense(64))
                            model.add(Dropout(0.2))
                            model.add(Dense(32))
                            model.add(Dense(1, activation = 'sigmoid'))
                            model.compile(optimizer=Adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
                            model.summary()
                            return model
                        
                        model = CNN()
                        #standard fit
                        model.fit([X_train1,X_train2], y_Train, validation_split=0.2, batch_size=batch_size, epochs = 10, verbose=2)
                        tmp_row = model.evaluate([X_train1,X_train2], y_Train, batch_size=batchs, verbose=1) # evaluate the model to get the loss and accuracy
                        tmp.append(tmp_row)

Results_table = np.vstack(tmp) 
Results_table = pd.DataFrame(Results_table, columns = [ "Loss", "accuracy"])                         
#%%
'''Use the data generator to fit the model.'''
model.fit_generator(generate_data_generator_for_two_images(X_train1, X_train2, y_Train), samples_per_epoch=len(X_Train), epochs=10)

#%% 
'''Image Plotting, f1 score & confusion matrix'''


#%%
'''Test data predict and csv export'''