# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv
"""
from matplotlib import animation
import torch
import Dataloader.transforms as transforms
from Dataloader.LPBF_loader_eval import LPBF_loader_eval
import torch.utils.data as data
from Trainer.pytorchtools import EarlyStopping
from Model.Network import ConvBayes
from Visualization.tSNE_plots import *
import os

#%%

def tsne_visualization(x_train, y_train, x_val, y_val, x_test, y_test, ckpt,opt,filename):
    
    
    folder_created = os.path.join('Figures/', filename)
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    
    # no augmentations used for linear evaluation
    transform_lineval = transforms.Compose([transforms.ToTensor()])

    train_set_lineval = LPBF_loader_eval(data=x_train, targets=y_train, transform=transform_lineval)
    val_set_lineval = LPBF_loader_eval(data=x_val, targets=y_val, transform=transform_lineval)
    test_set_lineval = LPBF_loader_eval(data=x_test, targets=y_test, transform=transform_lineval)

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval, batch_size=128, shuffle=True)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=128, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=128, shuffle=False)
    signal_length = x_train.shape[1]

    # loading the saved backbone
    backbone_lineval = ConvBayes(opt.feature_size).cuda()  # defining a raw backbone model
    

    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)

    print('Linear evaluation')
    
    backbone_lineval.eval()
    acc_trains = list()
    
    
    X, y = [], []
    
    for data, target in val_loader_lineval:
        
        data = data.cuda()
        output = backbone_lineval(data)
       
        X.append(output.detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())
        
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]
    
    
    train_embeddings = folder_created+'/'+str(filename)+'_embeddings'+ '.npy'
    train_labelsname = folder_created+'/'+str(filename)+'_labels'+'.npy'
    np.save(train_embeddings,X, allow_pickle=True)
    np.save(train_labelsname,y, allow_pickle=True)
    
    
    X, y = [], []
    
    for data, target in test_loader_lineval:
        
        data = data.cuda()
        output = backbone_lineval(data)
       
        X.append(output.detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())
        
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]
    
    
    train_embeddings = folder_created+'/'+str(filename)+'_test_embeddings'+ '.npy'
    train_labelsname = folder_created+'/'+str(filename)+'_test_labels'+'.npy'
    np.save(train_embeddings,X, allow_pickle=True)
    np.save(train_labelsname,y, allow_pickle=True)
    
    graph_name1= folder_created+'/'+str(filename)+'_2D'+'.png'
    graph_name2= folder_created+'/'+str(filename)+'_3D'+'.png'
    graph_name3= folder_created+'/'+str(filename)+'_3D'+'.gif'
    
    ax,fig,graph_name=TSNEplot(X,y,graph_name1,graph_name2,graph_name3,str(filename),limits=2.6,perplexity=20)
    
    
    return X, y,ax,fig,graph_name
 
