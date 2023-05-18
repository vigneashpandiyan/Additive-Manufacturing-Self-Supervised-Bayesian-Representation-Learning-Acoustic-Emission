# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:56:21 2023

@author: srpv
"""
import numpy as np
import pandas as pd


def normalize(ts,Features_train_max,Features_train_min):
    
    
    ts = 2. * (ts - Features_train_min) / (Features_train_max - Features_train_min) - 1.
    
    return ts

def load_LPBF(path, dataset_name,dataset_label):
    ##################
    # load raw data
    ##################
    
    print("dataset_path...",path)
    print("dataset_name...",dataset_name)
    
    Featurespace = np.load("{}/{}".format(path, dataset_name))
    classspace = np.load("{}/{}".format(path, dataset_label))
    
    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([Featurespace, classspace], axis=1)
    
    print("respective windows",data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())
    
    if minval >=6000:
        minval=6000
    else:
        minval=minval
    
    print("windows of the class: ",minval)
    
    data_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
    print("The dataset is well balanced: ",data_1.Categorical.value_counts())
    
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
        test_idx += target[:int(nb_samp * 0.25)]
        val_idx += target[int(nb_samp * 0.25):int(nb_samp * 0.5)]
        train_idx += target[int(nb_samp * 0.5):]

   
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

    # Process data
    x_test = x_test.reshape((-1, input_shape, 1))
    x_val = x_val.reshape((-1, input_shape, 1))
    x_train = x_train.reshape((-1, input_shape, 1))

    print("Train:{}, Test:{}, Class:{}".format(x_train.shape, x_test.shape, nb_class))

   

    return x_train, y_train, x_val, y_val, x_test, y_test, nb_class

def jitter(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)



def scaling(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def cutout(ts, perc=.1):
    seq_len = ts.shape[0]
    new_ts = ts.copy()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len-win_len-1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    # print("[INFO] start={}, end={}".format(start, end))
    new_ts[start:end, ...] = 0
    # return new_ts, ts[start:end, ...]
    return new_ts

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.3, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.3, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.7):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.3, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret
