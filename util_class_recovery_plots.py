# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:35:44 2017

@author: jlowe
"""
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.signal import savgol_filter
import seaborn as sns

class post_mortem:    
    '''Class for all post model scores and  plot'''
    def plot_calculations(self, model, scaler, test_X, train_X, train_y, test_y):
        '''Calculating variables to plot and RMS errors'''
        # make a prediction
        y_predicted_te = model.predict(test_X)
        test_X_r = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        y_predicted_te = np.concatenate((test_X_r[:,:],y_predicted_te), axis=1)
        y_predicted_te = scaler.inverse_transform(y_predicted_te)
        y_predicted_te = y_predicted_te[:,-1]
        # invert scaling for actual
        test_y_r = test_y.reshape((len(test_y), 1))
        test_y_inv = np.concatenate((test_X_r[:, :],test_y_r), axis=1)
        test_y_inv = scaler.inverse_transform(test_y_inv)
        test_y_inv = test_y_inv[:,-1]
        
        
        # make a prediction
        y_predicted_tr = model.predict(train_X)
        train_X_r = train_X.reshape((train_X.shape[0], train_X.shape[2]))
        # invert scaling for forecast
        y_predicted_tr = np.concatenate((train_X_r[:,:],y_predicted_tr), axis=1)
        y_predicted_tr = scaler.inverse_transform(y_predicted_tr)
        y_predicted_tr = y_predicted_tr[:,-1]
        # invert scaling for actual
        train_y_r = train_y.reshape((len(train_y), 1))
        train_y_inv = np.concatenate((train_X_r[:, :],train_y_r), axis=1)
        train_y_inv = scaler.inverse_transform(train_y_inv)
        train_y_inv = train_y_inv[:,-1]
     
        return train_y_inv, y_predicted_tr, test_y_inv, y_predicted_te
    
    def RMSE_tt(self, train_y_inv, test_y_inv, y_predicted_tr, y_predicted_te ):
        # calculate RMSE for test
        rmse_test = sqrt(mean_squared_error(test_y_inv, y_predicted_te))
        # calculate RMSE for train 
        rmse_train = sqrt(mean_squared_error(train_y_inv, y_predicted_tr))
        
        return rmse_train, rmse_test

    def plot_train_test(self, train_y_inv, y_predicted_tr, test_y_inv, y_predicted_te):
        '''Train & Test error plotting'''
        #Train
        rcParams['figure.figsize'] = 120, 10
        x_ax_tr = np.arange(0, len(y_predicted_tr), dtype=np.float)
        train_y_inv #=  savgol_filter(train_y_inv, 5, 2) # (data, window_length, polyorder)
        y_predicted_tr #= savgol_filter(y_predicted_tr, 5, 2)
        #Test
        x_ax_te = np.arange(len(y_predicted_tr), len(y_predicted_tr)+len(y_predicted_te), dtype=np.float)
        test_y_inv #= savgol_filter(test_y_inv, 5, 2) # (data, window_length, polyorder)
        y_predicted_te #= savgol_filter(y_predicted_te, 5, 2)
        #plt.scatter(x_ax_tr,inv_y_t)
        plt.plot(x_ax_tr,train_y_inv, color='r')
        plt.plot(x_ax_tr,y_predicted_tr, color='g')
        plt.plot(x_ax_te,test_y_inv, color='r')
        plt.plot(x_ax_te,y_predicted_te, color='b')
        plt.ylim(0.6,1)
        plt.show()
        
    def plot_calc_linear(self, pipeline, scaler, test_X, train_X, train_y, test_y):
        '''Calculating variables to plot and RMS errors'''
        # make a prediction
        y_predicted_te = pipeline.predict(test_X)
        y_predicted_te = y_predicted_te.reshape((y_predicted_te.shape[0],1))
        test_X_r = test_X.reshape((test_X.shape[0], test_X.shape[1]))
        # invert scaling for forecast
        y_predicted_te = np.concatenate((test_X_r[:,:],y_predicted_te), axis=1)
        y_predicted_te = scaler.inverse_transform(y_predicted_te)
        y_predicted_te = y_predicted_te[:,-1]
        # invert scaling for actual
        test_y_r = test_y.reshape((len(test_y), 1))
        test_y_inv = np.concatenate((test_X_r[:, :],test_y_r), axis=1)
        test_y_inv = scaler.inverse_transform(test_y_inv)
        test_y_inv = test_y_inv[:,-1]
        
        
        # make a prediction
        y_predicted_tr = pipeline.predict(train_X)
        y_predicted_tr = y_predicted_tr.reshape((y_predicted_tr.shape[0],1))
        train_X_r = train_X.reshape((train_X.shape[0], train_X.shape[1]))
        # invert scaling for forecast
        y_predicted_tr = np.concatenate((train_X_r[:,:],y_predicted_tr), axis=1)
        y_predicted_tr = scaler.inverse_transform(y_predicted_tr)
        y_predicted_tr = y_predicted_tr[:,-1]
        # invert scaling for actual
        train_y_r = train_y.reshape((len(train_y), 1))
        train_y_inv = np.concatenate((train_X_r[:, :],train_y_r), axis=1)
        train_y_inv = scaler.inverse_transform(train_y_inv)
        train_y_inv = train_y_inv[:,-1]
        
        return train_y_inv, y_predicted_tr, test_y_inv, y_predicted_te
    
    def correlation_plot(self, dataframe):
        dataframe.corr(method = 'spearman')
        np.set_printoptions(precision=2)
        sns.set(style="white")       
        # Generate a mask for the upper triangle
        mask = np.zeros_like(dataframe.corr(method = 'spearman') , dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Generate a custom diverging colormap
        cmap = sns.cubehelix_palette(n_colors=12, start=-2.25, rot=-1.3, as_cmap=True)
        rcParams['figure.figsize'] = 20, 20
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(dataframe.corr(method = 'spearman') ,annot=True,  mask=mask, cmap=cmap, vmax=.3, square=True)
        plt.show()
     