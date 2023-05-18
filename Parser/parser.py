# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:09:53 2023

@author: srpv
"""

import argparse

def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--K', type=int, default=16, help='Number of augmentation for each sample') # Bigger is better.

    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    
   
    parser.add_argument('--patience', type=int, default=400,
                        help='training patience')
    
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')
    
    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')
    
    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    

    # Testing parameters
    parser.add_argument('--learning_rate_test', type=float, default=0.01,
                        help='learning_rate_test')
    
    parser.add_argument('--patience_test', type=int, default=100,
                        help='number of training patience')
    
    
    #Training and Testing
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    
    parser.add_argument('--bayesian_train_size', type=int, default=1,
                        help='bayesian_train_size') #default training

    parser.add_argument('--epochs_test', type=int, default=100,
                        help='number of test epochs')
    
    # parser.add_argument('--bayesian_size', type=int, default=1,
    #                     help='bayesian_size')
    
    
    opt = parser.parse_args()
    return opt

