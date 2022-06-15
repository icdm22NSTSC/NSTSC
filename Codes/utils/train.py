# -*- coding: utf-8 -*-


from utils.data import * 
from utils.train_utils import * 

# Train a tree phase classifier 
def train_model(did, dataset_path_, dname, epochs = 100, \
            normalize_timeseries = True, lr = 0.1):
    
    Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw \
        = Readdataset(dataset_path_, dname, normalize_timeseries)
    N, T = calculate_dataset_metrics(Xtrain_raw)
    intvlen, nintv = Get_intinfo(T)
    
    Xtrain_raw, Xtrain_fft, Xtrain_derv = Splitview(Xtrain_raw, T)
    Xval_raw, Xval_fft, Xval_derv = Splitview(Xval_raw, T)
    Xtest_raw, Xtest_fft, Xtest_derv = Splitview(Xtest_raw, T)
    
    Xtrain_raw, Xtrain_fft, Xtrain_derv, Xval_raw, Xval_fft, Xval_derv, \
    Xtest_raw, Xtest_fft, Xtest_derv = Extract_intfea(Xtrain_raw, Xtrain_fft, \
    Xtrain_derv, Xval_raw, Xval_fft, Xval_derv, Xtest_raw, Xtest_fft, \
        Xtest_derv, nintv, intvlen)
    
    Xtrain = np.concatenate((Xtrain_raw, Xtrain_fft, Xtrain_derv), 1)
    Xval = np.concatenate((Xval_raw, Xval_fft, Xval_derv), 1)
    Xtest = np.concatenate((Xtest_raw, Xtest_fft, Xtest_derv), 1)
    
    Xtrain, Xval, Xtest = Stand_data(Xtrain, Xval, Xtest)
    
    N, T = calculate_dataset_metrics(Xtrain)
    classnum = int(np.max(ytrain_raw) + 1)
    
    Tree = Build_tree(Xtrain, Xval, ytrain_raw, yval_raw, epochs, classnum, \
                   learnrate = lr, savepath = './utils/')
    
    return Tree
      




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    