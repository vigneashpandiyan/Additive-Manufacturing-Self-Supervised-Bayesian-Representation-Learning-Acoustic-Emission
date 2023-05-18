# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:54:48 2023

@author: srpv
"""

import numpy as np
import torch.utils.data as data


class LPBF_Inter_Intra_reasoning(data.Dataset):

    def __init__(self, data, targets, K, transform, transform_cut, totensor_transform):

        
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.transform_cut = transform_cut
        self.totensor_transform = totensor_transform

    def __getitem__(self, index):
       
        ts, target = self.data[index], self.targets[index]
        
        ts_list = list()
        ts_list0 = list()
        ts_list1 = list()
        label_list = list()

        if self.transform is not None:
            for _ in range(self.K):
                ts_transformed = self.transform(ts.copy())
                ts_cut0, ts_cut1, label = self.transform_cut(ts_transformed)
                ts_list.append(self.totensor_transform(ts_transformed))
                ts_list0.append(self.totensor_transform(ts_cut0))
                ts_list1.append(self.totensor_transform(ts_cut1))
                label_list.append(label)

        return ts_list, ts_list0, ts_list1, label_list, target

    def __len__(self):
        return self.data.shape[0]