# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:56:21 2023

@author: srpv
"""

import numpy as np
import torch.utils.data as data



class LPBF_loader_eval(data.Dataset):

    def __init__(self, data, targets, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __getitem__(self, index):
        ts, target = self.data[index], self.targets[index]
        
        # ts_mean = np.max(ts)
        # ts_std = np.std(ts)
        # ts = (ts - ts_mean) / ts_std
        
      
        
        if self.transform is not None:
            ts_transformed = self.transform(ts.copy())
        else:
            ts_transformed = ts

        return ts_transformed, target

    def __len__(self):
        return self.data.shape[0]

