# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import os
import torch


def setup_seed(seed):
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)


# Shuffle data
def Shuffle(X, y):
    nums = list(range(len(y)))
    random.shuffle(nums)
    random.shuffle(nums)
    random.shuffle(nums)
    X = X[nums,:]
    y = y[nums]
    return X, y


# Load Dataset given a dataset's path
def Readdataset(dataset_path_, dataset_name, standalize, val = False):
    path_up = './'
    data_path = path_up + dataset_path_ + '/'
    Xtrain = np.load(data_path + dataset_name + 'Xtrain.npy')
    Xtest = np.load(data_path + dataset_name + 'Xtest.npy')
    
    ytrain = np.load(data_path + dataset_name + 'ytrain.npy')
    ytest = np.load(data_path + dataset_name + 'ytest.npy')
    
    Xtrain, ytrain = Shuffle(Xtrain, ytrain)
    Xtest, ytest = Shuffle(Xtest, ytest)
    
    Ntrain = Xtrain.shape[0]
    Xall, yall = np.concatenate((Xtrain, Xtest)), np.concatenate((ytrain, ytest))
        
    yset = np.array(list(set(yall))).astype(int)
    classnum = len(yset)    
    for ci in range(classnum):
        yall[yall == yset[ci]] = ci
    
    ss = StandardScaler()
    if standalize:
        Xall = ss.fit_transform(Xall)
        
    Xall_fft = np.fft.fft(Xall)
    Xall_fft = np.abs(Xall_fft)
    Xall_dif = Xall[:,1:] - Xall[:,:-1]
    Xall_dif = np.concatenate((Xall_dif[:,0].reshape([-1,1]), Xall_dif), 1)
    Xall = np.concatenate((Xall, Xall_fft, Xall_dif), 1)
    if standalize:
        Xall = ss.fit_transform(Xall)
    Xtrain, Xtest = Xall[:Ntrain,:], Xall[Ntrain:,:] 
    ytrain, ytest = yall[:Ntrain,], yall[Ntrain:,]
    
    if val:
        Ntest = Xtest.shape[0]
        Nval = int(Ntest * 0.5)
        Xval, yval = Xtest[:Nval, :], ytest[:Nval,]
        Xtest, ytest = Xtest[Nval:, :], ytest[Nval:,]
    else:
        Xval = Xtest - 0
        yval = ytest - 0
    
    return Xtrain, ytrain, Xval, yval, Xtest, ytest


# Dimension of data
def calculate_dataset_metrics(Xtrain):
    
    N, T = Xtrain.shape[0], int(Xtrain.shape[1]/3)
    
    return N, T

# Compute interval length
def Get_intinfo(T):
    
    if T > 40:
        nintv = 20
        intvlen = int(T//nintv)
    else:
        intvlen = 5
        nintv = int(T//intvlen)
    
    return intvlen, nintv


# Multi-view representation
def Splitview(X, T):
    
    Xori = X[:,:T]
    Xfft = X[:,T:2*T]
    Xspe = X[:,2*T:]
    
    return Xori, Xfft, Xspe


# Interval feature extraction
def Extract_intfea(Xtrain_raw, Xtrain_fft, Xtrain_derv, Xval_raw, \
                   Xval_fft, Xval_derv, Xtest_raw, Xtest_fft, Xtest_derv,\
                   nintv, intvlen):
    
    Xtrain_raw = Addstatfea(Xtrain_raw, nintv, intvlen)
    Xtrain_fft = Addstatfea(Xtrain_fft, nintv, intvlen)
    Xtrain_derv = Addstatfea(Xtrain_derv, nintv, intvlen)
    Xval_raw = Addstatfea(Xval_raw, nintv, intvlen)
    Xval_fft = Addstatfea(Xval_fft, nintv, intvlen)
    Xval_derv = Addstatfea(Xval_derv, nintv, intvlen)
    Xtest_raw = Addstatfea(Xtest_raw, nintv, intvlen)
    Xtest_fft = Addstatfea(Xtest_fft, nintv, intvlen)
    Xtest_derv = Addstatfea(Xtest_derv, nintv, intvlen)
    
    return Xtrain_raw, Xtrain_fft, Xtrain_derv, Xval_raw, Xval_fft, \
        Xval_derv, Xtest_raw, Xtest_fft, Xtest_derv


# Add statistical features from interval data
def Addstatfea(X, n, t):
    X = Addmean(X, n, t)
    X = Addstd(X, n, t)
    X = Addmin(X, n, t)
    X = Addmax(X, n, t)
    X = Addmedian(X, n, t)
    X = AddIQR(X, n, t)
    X = Addslope(X, n, t)
    return X


# Mean feature
def Addmean(X, n, t):
    T = X.shape[1]
    Xmean = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmean[:,i] = np.mean(X[:,i*t:(i+1)*t],1)
       else:
           Xmean[:,i] = np.mean(X[:,i*t:T],1)
    X = np.concatenate((X, Xmean),1)
    return X


# Std feature
def Addstd(X, n, t):
    T = X.shape[1]
    Xstd = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xstd[:,i] = np.std(X[:,i*t:(i+1)*t],1)
       else:
           Xstd[:,i] = np.std(X[:,i*t:T],1)
    X = np.concatenate((X, Xstd),1)
    return X


# Min feature
def Addmin(X, n, t):
    T = X.shape[1]
    Xmin = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmin[:,i] = np.min(X[:,i*t:(i+1)*t],1)
       else:
           Xmin[:,i] = np.min(X[:,i*t:T],1)
    X = np.concatenate((X, Xmin),1)
    return X


# Max feature
def Addmax(X, n, t):
    T = X.shape[1]
    Xmax = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmax[:,i] = np.max(X[:,i*t:(i+1)*t],1)
       else:
           Xmax[:,i] = np.max(X[:,i*t:T],1)
    X = np.concatenate((X, Xmax),1)
    return X


# Median feature
def Addmedian(X, n, t):
    T = X.shape[1]
    Xmedian = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmedian[:,i] = np.median(X[:,i*t:(i+1)*t],1)
       else:
           Xmedian[:,i] = np.median(X[:,i*t:T],1)
    X = np.concatenate((X, Xmedian),1)
    return X


# IQR feature
def AddIQR(X, n, t):
    T = X.shape[1]
    XIQR = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           xtem = np.percentile(X[:,i*t:(i+1)*t],[25,75], 1).T
           XIQR[:,i] = xtem[:,1] - xtem[:,0]        
       else:
           xtem = np.percentile(X[:,i*t:T],[25,75], 1).T
           XIQR[:,i] = xtem[:,1] - xtem[:,0]    
    X = np.concatenate((X, XIQR),1)
    return X


# Slope feature
def Addslope(X, n, t):
    T = X.shape[1]
    Xslope = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           intlen = t
           p = np.array(range(1,intlen+1)).reshape([1,-1])
           xtem = X[:,i*t:(i+1)*t]
           slopecur = (np.matmul(p,xtem.T)-np.sum(p)*(np.mean(xtem,1)))/\
               (np.matmul(p,p.T)-np.sum(p)*np.mean(p))
           Xslope[:,i] = np.arctan(slopecur)   
       else:
           intlen = T - i*t + 1
           p = np.array(range(1,intlen+1)).reshape([1,-1])
           xtem = X[:,i*t:T]
           slopecur = (p*xtem.T-sum(p)*(np.mean(xtem,1)))/(p*p.T-sum(p)*\
                                                           np.mean(p))
           Xslope[:,i] = np.arctan(slopecur)   
    X = np.concatenate((X, Xslope),1)
    return X


# Standardize data
def Stand_data(Xtrain, Xval, Xtest, val = False):
    if val:
        Ntrain = Xtrain.shape[0]
        Nval = Xval.shape[0]
        Xall = np.concatenate((Xtrain, Xval, Xtest), 0)
        ss = StandardScaler()
        Xall = ss.fit_transform(Xall)
        Xtrain = Xall[:Ntrain,:]
        Xval = Xall[Ntrain:Ntrain+Nval, :]
        Xtest = Xall[Ntrain+Nval:,:]
        
    else:
        Ntrain = Xtrain.shape[0]
        Xall = np.concatenate((Xtrain, Xtest),0)
        ss = StandardScaler()
        Xall = ss.fit_transform(Xall)
        Xtrain = Xall[:Ntrain,:]
        Xtest = Xall[Ntrain:,:]
        Xval = Xtest

    return Xtrain, Xval, Xtest



