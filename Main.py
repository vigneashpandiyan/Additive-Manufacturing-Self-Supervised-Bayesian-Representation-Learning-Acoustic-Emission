# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:29:36 2023

@author: srpv

This repo hosts the codes that were used in journal work: 
"Self-Supervised Bayesian Representation Learning of Acoustic Emissions 
from Laser Powder Bed Fusion Process for In-situ Monitoring"

--> Cite the following work

"Self-Supervised Bayesian Representation Learning of Acoustic Emissions 
from Laser Powder Bed Fusion Process for In-situ Monitoring"

@contact --> vigneashwara.solairajapandiyan@empa.ch

"""
import torch
torch.cuda.is_available()
import pandas as pd
import os
import matplotlib.animation as animation

from Parser.parser import parse_option
from Dataloader.AE_dataset_loader import *
from Trainer.Bayesian_selftrain import Bayesian_backbone,count_parameters
from Model.Network import ConvBayes
from Evaluation.Backbone_evaluation import Bayes_backbone_evaluation
from Visualization.tSNE import tsne_visualization
from Visualization.Bayesian_latent_plots import latent_plots
from Generalization.Backbone_generalization import model_generalization


from Transfer_Learning.TL import transfer_learning
from Classifiers.Classification import classifier_ML
from Anomaly_detection.Utils_one_class_SVM import anomaly_detection


#%%
Seeds = [0, 1, 2, 3, 4]
for seed in Seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)

#%%
'''
Download the dataset from following link
-- https://polybox.ethz.ch/index.php/s/9xLsB1kJORolfTc
'''

path=r'C:\Users\srpv\Desktop\LPBF Bayesian Self-Time learning\Data' #place the data inside the datafolder
dataset_name='D1_rawspace_5000.npy'
dataset_label= 'D1_classspace_5000.npy'
opt = parse_option()
print(opt.class_type)

#%%
#Augumentation type
aug1 = ['cutout']
aug2 = ['jitter']
aug3 = ['scaling']
aug4 = ['magnitude_warp']
aug5 = ['time_warp']
aug6 = ['window_slice']
aug7 = ['window_warp']


opt.aug_type = aug1 + aug2 + aug3 + aug4 + aug5 + aug6 + aug7      
print(opt.aug_type)

#%%
#Loading dataset
x_train, y_train, x_val, y_val, x_test, y_test, nb_class= load_LPBF(path, dataset_name,dataset_label)

#%%
# Model training
model = Bayesian_backbone(x_train, y_train, opt,'Clustering_D1')
backbone = ConvBayes(opt.feature_size).cuda()
# Model parameters
count_parameters(model)
count_parameters(backbone)

#%%

folder_created = os.path.join('Figures/', 'Clustering_D1')
print(folder_created)
ckpt ='{}/backbone_best.tar'.format(folder_created)
lkpt ='{}/Clustering_D1_linear.tar'.format(folder_created)

acc_test, epoch_max_point = Bayes_backbone_evaluation(
                x_train, y_train, x_val, y_val, x_test, y_test,nb_class,ckpt,
                opt,'Clustering_D1',None)
    #tsne-viz
X, y,ax,fig,graph_name= tsne_visualization(x_train, y_train, x_val, y_val, x_test, y_test, ckpt,opt,filename='Clustering_D1')

def rotate(angle):
      ax.view_init(azim=angle)
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))

latent_plots('Clustering_D1',)


#%% Back-bone representation

model_generalization('D2',path,ckpt,opt)
model_generalization('D3',path,ckpt,opt)
model_generalization('D4',path,ckpt,opt)


#%% Transfer Learning

acc_test, epoch_max_point =transfer_learning('D2',path,ckpt,lkpt,opt)
acc_test, epoch_max_point =transfer_learning('D3',path,ckpt,lkpt,opt)
acc_test, epoch_max_point =transfer_learning('D4',path,ckpt,lkpt,opt)

#%% Pointing to the self-supervised bayesian backbone

path=r'C:\Users\srpv\Desktop\LPBF Bayesian Self-Time learning\Figures'

#%% Anomaly detection

anomaly_detection('D1',path)
anomaly_detection('D2',path)
anomaly_detection('D3',path)
anomaly_detection('D4',path)


#%% Classifier

classifier_ML(path, 'Clustering_D1')
classifier_ML(path, 'Clustering_D2')
classifier_ML(path, 'Clustering_D3')
classifier_ML(path, 'Clustering_D4')

#%%









