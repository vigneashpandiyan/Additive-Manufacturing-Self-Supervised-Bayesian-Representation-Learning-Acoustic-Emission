# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:29:36 2023

@author: srpv
"""
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import joblib
from Classifiers.Helper import *
from Classifiers.plot_roc import *

from sklearn.linear_model import LogisticRegression

#%%

def LR(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder):
    
    plt.rcParams.update(plt.rcParamsDefault)

    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train,y_train)
    
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model,X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))  
    
    predictions = model.predict(X_test)
    print("LogisticRegression Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    
    
    graph_name1= 'LR'+'_without normalization w/o Opt'
    graph_name2=  'Logistic Regression'
    
    graph_1= folder+'LR'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= folder+'LR'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        
        plt.figure(figsize = (20, 10),dpi=200)
        ConfusionMatrixDisplay.from_predictions(y_test, predictions,normalize=normalize,cmap=plt.cm.RdPu)
        plt.title(title, size = 12)
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        
        
    savemodel= folder+ 'LR'+'_model'+'.sav'    
    joblib.dump(model, savemodel)
    
    Title1= folder+'LR'+'_Roc'+'.png'
    Title2=folder+ 'LR'+'_Precision_Recall'+'.png'
    plot_roc(model,Featurespace,classspace,classes,Title1,Title2)