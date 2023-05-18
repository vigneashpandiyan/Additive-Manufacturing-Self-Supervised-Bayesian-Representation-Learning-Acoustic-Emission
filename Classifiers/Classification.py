# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:29:36 2023

@author: srpv
"""
import numpy as np
import pandas as pd
import os     


from Classifiers.RF import *
from Classifiers.SVM import *
from Classifiers.XGBoost import *
from Classifiers.NeuralNets import *
from Classifiers.Logistic_regression import *
from Classifiers.Helper import *
from Classifiers.plot_roc import *

#%%

def classifier_ML(path, dataset_name):
    
    folder=dataset_name +'_'+'ML'
    
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    
    print("dataset_path...",path)
    print("dataset_name...",dataset_name)
    
    
    X_train=str(dataset_name)+'_embeddings'+ '.npy'
    y_train=str(dataset_name)+'_labels'+'.npy'
    
    X_train = np.load("{}/{}/{}".format(path, dataset_name,X_train))
    X_train = pd.DataFrame(X_train)
    
    y_train = np.load("{}/{}/{}".format(path, dataset_name,y_train))
    y_train = pd.DataFrame(y_train)
    y_train.columns = ['Categorical']
    classes=np.unique(y_train)
    classes = list(classes)
    
    num_cols = len(list(X_train))
    rng = range(1, num_cols + 1)
    Featurenames = ['Feature_' + str(i) for i in rng] 
    X_train.columns = Featurenames
    feature_cols=list(X_train.columns) 
    X_train.info()
    X_train.describe()
    X_train.head()


      
    X_test=str(dataset_name)+'_test_embeddings'+ '.npy'
    y_test=str(dataset_name)+'_test_labels'+'.npy'
    
    X_test = np.load("{}/{}/{}".format(path, dataset_name,X_test))
    X_test = pd.DataFrame(X_test)
    
    y_test = np.load("{}/{}/{}".format(path, dataset_name,y_test))
    y_test = pd.DataFrame(y_test)
    y_train.columns = ['Categorical']
   
    
    
    num_cols = len(list(X_test))
    rng = range(1, num_cols + 1)
    Featurenames = ['Feature_' + str(i) for i in rng] 
    X_test.columns = Featurenames
    feature_cols=list(X_test.columns) 
    X_test.info()
    X_test.describe()
    X_test.head()


    Featurespace=X_test
    classspace=y_test
    # X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.25, random_state=66)

    

    RF(X_train, X_test, y_train, y_test,100,feature_cols,Featurespace, classspace,classes,folder)
    SVM(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
    LR(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
    XGBoost(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
    NN(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
   
    
