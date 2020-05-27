# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:39:11 2017

@author: jlowe
"""
import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor #sklearn wrapper for XGBoost
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense
from keras import regularizers, optimizers

from util_preprocessing import Denoise, predict_IAngle, Incident_Angle_scaling, dx_dy, data_wrangling, Scaling
np.random_seed=(9)

#%%
'''Import the Data'''
#Load data
os.chdir("C:/Users/jlowe/Documents/Projects/Kaggle - statoil iceberg prj/")
train = pd.read_json("train.json")
train_df = train
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
    X = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , ((x_band1/(x_band2+0.000000001)))[:, :, :, np.newaxis]], axis=-1) #also adding a term to make it a 3 channel image
    X_angle = np.array(df.inc_angle)
    if y == True:
        y = np.array(df["is_iceberg"])
        return X, X_angle, y
    else:
        return X, X_angle

#%%
'''Feature normalisation'''
def Scaling(X):
    scaler_band1 = MinMaxScaler(feature_range=(0, 1))
    #Apply the transforms and reshape to original shape
    Xtmp = scaler_band1.fit_transform(X.reshape((X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])))   
    Xtmp1 = Xtmp.reshape((X.shape[0],75,75,3))
    return Xtmp1

#%%
'''Pipeline'''
#Denoising
train1 = Denoise(train, filter_strength=10)
#Adding features
#train2 = dx_dy(train, ksize=5)
#Calculate missing Incident Angles
train3 = predict_IAngle(train1)
#Apply scaler based on Incident Angle of aquisition.
train4 = Incident_Angle_scaling(train3, scaler=0.015)
#Data wrangling
X_Train, X_Angle_Train, y_Train = data_wrangling(train4, y=True)
#scaling
X_Train5 = Scaling(X_Train)

X_Train6 = np.pad(X_Train5, pad_width=((0, 0), (75, 74), (75, 74), (0, 0)), mode="constant") #(0, 0), (75, 74), (75, 74), (0, 0))
##Image augmentation
#gen_flow = Image_Augmentation(X_Train, X_Angle_Train, y_Train)
#%%
'''Image Augmentation'''
batch_gen_size = 64
# Define the image transformations here
datagen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 40)
#    return gen_flow
datagen.fit(X_Train)

datagen.flow(X_Train, y_Train, batch_size=batch_gen_size, save_to_dir='C:/Users/jlowe/Box Sync/Projects/Kaggle - statoil iceberg prj/Augmented_Images')
#%%
'''Train & Test Split'''
#The images may need to be randomly shuffled at the top
Cut = 1605
X_Train6=X_Train6[:,:,:,0:3]
X_train, X_test = X_Train5[:Cut,:,:,:], X_Train5[Cut:,:,:,:]
y_train, y_test = y_Train[:Cut,], y_Train[Cut:,]

#%%
'''Convolutional Neural Network'''

batch_size = 2


reg = regularizers.l2(0.001)
input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])

ep = 20#0, 200, 400, 800]#200
batch_size = 2
learnrate = 0.001#, 0.0006, 0.0001, 0.00006, 0.00001, 0.000006, 0.000001, 0.0000006, 0.0000001]
l2 = 0.00001#, 0.0006, 0.0001, 0.00006, 0.00001, 0.000006, 0.000001, 0.0000006, 0.0000001]
conv_layer = 4
Drop = 0
count = 0


#Define parameters
reg = regularizers.l2(l2)
Adam = optimizers.Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

from keras.applications.resnet50 import ResNet50
from keras.models import Model

model = ResNet50(weights = 'imagenet')
# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalMaxPooling2D()(x)
# let's add a fully-connected layer
x = Dense(124, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics = ['accuracy'])

# train the model on the new data for a few epochs
model.fit(X_Train6, y_Train, validation_split=0.05, batch_size=batch_size, epochs = ep, verbose=1)
#%%
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics = ['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(X_Train6, y_Train, validation_split=0.05, batch_size=batch_size, epochs = ep, verbose=1)

#standard fit
history = model.fit(X_train, y_train, validation_split=0.05, batch_size=batch_size, epochs = ep, verbose=1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#%%
#epoch = 16
#history = model.fit(X_train, y_train, validation_split=0.05, batch_size=batch_size, epochs = ep, verbose=1)
## plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()
##%%
##Fiting to datagen
#history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_gen_size),samples_per_epoch=len(X_train)*batch_gen_size, validation_data=datagen.flow(X_test, y_test), validation_steps=(len(X_test)),  epochs=20)
## plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()
#%% 
'''Image Plotting, f1 score & confusion matrix'''
#import seaborn as sns
#from sklearn.metrics import confusion_matrix
#
##Correlation Matrix
##First make train and test predictions
#y_train_pred = np.round(model.predict(X_train))
#y_test_pred = np.round(model.predict(X_test))
#
#corr_train = confusion_matrix(y_train.reshape((y_train.shape[0],1)), y_train_pred)
#corr_test = confusion_matrix(y_test.reshape((y_test.shape[0],1)), y_test_pred)
##%%
#from pylab import rcParams
#rcParams['figure.figsize'] = 12, 5
#fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
#sns.heatmap(corr_train, ax=ax1, annot=True, xticklabels= "S" "I",yticklabels= "S" "I",  cmap="YlGnBu")#,  vmin=0, vmax=750)
#sns.heatmap(corr_test, ax=ax2, annot=True, xticklabels= "S" "I",yticklabels= "S" "I",  cmap="YlGnBu")
#ax1.set_title('Train Data')
#ax2.set_title('Test Data')

#%%
test1 = pd.read_json("test.json")
test1.inc_angle = test1.inc_angle.replace('na', 0)
test1.inc_angle = test1.inc_angle.astype(float).fillna(0.0)
#%%
test = test1
ID = test['id']
ID = np.array(ID)

'''Pipeline'''
#Denoising
test1 = Denoise(test, filter_strength=10)
#Adding features
#test2 = dx_dy(test, ksize=5)
#Calculate missing Incident Angles
test3 = predict_IAngle(test1)
#Apply scaler based on Incident Angle of aquisition.
test4 = Incident_Angle_scaling(test3, scaler=0.015)
#Data wrangling
X_test, X_Angle_test = data_wrangling(test4, y=False)
#scaling
X_test5 = Scaling(X_test)
X_test5=X_test5[:,:,:,0:2]
X_test5 = np.pad(X_test5, pad_width=((0, 0), (63, 63), (63, 63), (0, 0)), mode="constant")
##Image augmentation
#gen_flow = Image_Augmentation(X_Train, X_Angle_Train, y_Train)
y_pred = model.predict(X_test5, batch_size=batch_size, verbose=1)

sub01 = np.concatenate((np.reshape(ID, (ID.shape[0],1)), np.array(y_pred)),axis=1)
sub01 = pd.DataFrame(sub01, columns = [ "id", "is_iceberg"]) 

sub01.to_csv("sub01", sep=',', index = False)
#%%
import seaborn as sns;sns.set(color_codes=True)
#
#ax1 = sns.distplot(np.asarray(test3.iloc[1000,1]))
#ax1 = sns.distplot(np.asarray(train3.iloc[1000,1]))
#%%
ax2 = sns.distplot(X_test5[:,:,:,0].flatten())
ax2 = sns.distplot(X_Train5[:,:,:,0].flatten())
#%%
y_pred_tr = model.predict(X_Train5, batch_size=batch_size, verbose=1)
ax2 = sns.distplot(y_pred[:,:].flatten())
ax2 = sns.distplot(y_pred_tr[:,:].flatten())

