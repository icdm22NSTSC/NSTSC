# -*- coding: utf-8 -*-
"""
Source code for "Neuro-symbolic Models for Interpretable Time Series 
Classification using Temporal Logic Description"
"""

import numpy as np
import torch,random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.Models import *
from utils.train import * 
from utils.data import *


    
setup_seed(20)

def main():
    Filepath = '../MiceTrainTest/'
    dataset_map = [("D10", 1)]
    normalize_dataset = True
    Max_epoch = 100
    
    for dname, did in dataset_map:
        print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)
        dataset_name_ = Filepath + dname
        Tree = train_model(did, dataset_name_, dname, epochs=Max_epoch, \
                    normalize_timeseries=normalize_dataset)
            
        
    
    

        
if __name__ == "__main__":
    main()

