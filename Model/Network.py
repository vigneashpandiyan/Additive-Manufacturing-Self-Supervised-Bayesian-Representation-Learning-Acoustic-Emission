# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:56:21 2023

@author: srpv
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from blitz.modules import BayesianLinear,BayesianConv1d

sys.path.append("..")

class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print(x.shape)
        return x

class ConvBayes(nn.Module): 
    def __init__(self,feature_size):
        super(ConvBayes, self).__init__()
        self.feature_size = feature_size
        self.dropout=0.1
        
        
        self.Conv1d1 = BayesianConv1d(in_channels=1, out_channels=4, kernel_size=8)
        self.Conv1d2 = BayesianConv1d(in_channels=4, out_channels=8, kernel_size=8)
        self.Conv1d3 = BayesianConv1d(in_channels=8, out_channels=16, kernel_size=8)
        self.Conv1d4 = BayesianConv1d(in_channels=16, out_channels=32, kernel_size=8)
        self.Conv1d5 = BayesianConv1d(in_channels=32, out_channels=self.feature_size, kernel_size=8)
        
        # self.blinear1 = BayesianLinear(64*13, 64)
        # self.blinear2 = BayesianLinear(64, 32)
        # self.blinear3 = BayesianLinear(32, emb_dim)
        self.flatten = torch.nn.Flatten()
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer(),
        )
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer(),
        )
        
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer(),
         
        )
        
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer(),
        )

        self.conv5 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            torch.nn.AdaptiveAvgPool1d(1),
            PrintLayer(),
        )
        
     
    def forward(self, x):
        
       
        x = x.view(x.shape[0], 1, -1)
        # print("into the network_view",x.shape)
        
        x=self.Conv1d1(x)
        x = self.conv1(x)
        
        x=self.Conv1d2(x)
        x = self.conv2(x)
        
        x=self.Conv1d3(x)
        x = self.conv3(x)
        
        x=self.Conv1d4(x)
        x = self.conv4(x)
        
        x=self.Conv1d5(x)
        x = self.conv5(x)
        
      
        x = self.flatten(x)
       
        x = F.normalize(x, dim=1)
        
        return x

class LinearBayes(nn.Module): 
    def __init__(self,nb_class,feature):
        super(LinearBayes, self).__init__()
        self.feature = feature
        self.dropout=0.1
        self.nb_class=nb_class
        
        # self.fc1=BayesianLinear(64, 32)
        # self.fc2=BayesianLinear(32, 3)
        
        self.fc1=nn.Linear(64, 32)
        self.fc2=nn.Linear(32, 3)
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
           )
        
        
    def forward(self, x):
    
        x=self.fc1(x)
        x = self.conv1(x)
        x=self.fc2(x)
        
        return x