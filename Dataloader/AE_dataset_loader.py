# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:56:21 2023

@author: srpv
"""

import numpy as np
import pandas as pd

#%%

def normalize(ts,Features_train_max,Features_train_min):
    
    ts = 2. * (ts - Features_train_min) / (Features_train_max - Features_train_min) - 1.
    
    return ts

def load_LPBF(path, dataset_name,dataset_label):
    ##################
    # load raw data
    ##################
    
    print("Dataset path...",path)
    print("Dataset name...",dataset_name)
    
    Featurespace = np.load("{}/{}".format(path, dataset_name))
    classspace = np.load("{}/{}".format(path, dataset_label))
    
    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([Featurespace, classspace], axis=1)
    
    print("Respective windows per category",data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())
    
   
    minval=np.round(minval,decimals=-3)    
    print("windows of the class: ",minval)
    
    
    data_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
    print("Balanced dataset: ",data_1.Categorical.value_counts())
    
    data=data_1.iloc[:,:-1]
    label=data_1.iloc[:,-1]
    
    x = data.to_numpy() 
    y = label.to_numpy() 
    
    input_shape = x.shape[1]
    nb_class = np.unique(y)
    
    print("Unique classes in the dataset [LoF, Conduction, Keyhole] ",len(nb_class))
    ############################################
    # Combine all train and test data for resample
    ############################################

    ts_idx = list(range(x.shape[0]))
    np.random.shuffle(ts_idx)
    x_all = x[ts_idx]
    y_all = y[ts_idx]
    
    Features_train_max = np.max(x_all)
    Features_train_min = np.min(x_all)
    label_idxs = np.unique(y_all)
    
    
    test_idx = []
    val_idx = []
    train_idx = []
    
    for idx in label_idxs:
        target = list(np.where(y_all == idx)[0])
        nb_samp = int(len(target))
        test_idx += target[:int(nb_samp * 0.30)]
        val_idx += target[int(nb_samp * 0.30):int(nb_samp * 0.40)]
        train_idx += target[int(nb_samp * 0.60):]

   
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
   


    print("[Stat] Whole dataset: mean={}, std={}".format(np.mean(x_all),np.std(x_all)))
    print("[Stat] Train class: mean={}, std={}".format(np.mean(x_train),np.std(x_train)))
    print("[Stat] Val class: mean={}, std={}".format(np.mean(x_val), np.std(x_val)))
    print("[Stat] Test class: mean={}, std={}".format(np.mean(x_test), np.std(x_test)))
    
    
    x_train=normalize(x_train,Features_train_max,Features_train_min)
    x_val=normalize(x_val,Features_train_max,Features_train_min)
    x_test=normalize(x_test,Features_train_max,Features_train_min)
    

    print("[Stat-normalize] Train class: mean={}, std={}".format(np.mean(x_train),np.std(x_train)))
    print("[Stat-normalize] Val class: mean={}, std={}".format(np.mean(x_val), np.std(x_val)))
    print("[Stat-normalize] Test class: mean={}, std={}".format(np.mean(x_test), np.std(x_test)))

    # reshaping the data
    x_test = x_test.reshape((-1, input_shape, 1))
    x_val = x_val.reshape((-1, input_shape, 1))
    x_train = x_train.reshape((-1, input_shape, 1))

    print("Train:{}, Test:{},Val:{} ,Class:{}".format(x_train.shape, x_test.shape , x_val.shape, nb_class))

   
    return x_train, y_train, x_val, y_val, x_test, y_test, nb_class

