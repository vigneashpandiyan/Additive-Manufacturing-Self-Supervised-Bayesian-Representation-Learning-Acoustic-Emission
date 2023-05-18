# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:29:36 2023

@author: srpv
"""
from Dataloader.AE_dataset_loader import load_LPBF
from Evaluation.Backbone_evaluation import Bayes_backbone_evaluation
from Visualization.tSNE import tsne_visualization
import matplotlib.animation as animation
import numpy as np

def model_generalization(dataset,path,ckpt,opt):
    
    
    graph_name='Clustering_'+str(dataset)
    
    dataset_name=str(dataset)+'_rawspace_5000.npy'
    dataset_label= str(dataset)+'_classspace_5000.npy'
    x_train, y_train, x_val, y_val, x_test, y_test, nb_class = load_LPBF(path,dataset_name,dataset_label)
    
    
    acc_test, epoch_max_point = Bayes_backbone_evaluation(
                    x_train, y_train, x_val, y_val, x_test, y_test,nb_class,ckpt,
                    opt,graph_name, None)
    
       
    X, y,ax,fig,graph_name= tsne_visualization(x_train, y_train, x_val, y_val, x_test, y_test, ckpt,opt,filename=graph_name) #'Clustering_20micron'
    angle = 3
    def rotate(angle):
          ax.view_init(azim=angle)
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))