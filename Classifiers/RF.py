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
from sklearn.ensemble import RandomForestClassifier

#%%

def RF(X_train, X_test, y_train, y_test,n_estimators,feature_cols,Featurespace, classspace,classes,folder):
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    model = RandomForestClassifier(n_estimators=n_estimators , oob_score=True)
    model.fit(X_train, y_train)
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model,X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))       
    #Accuracy of the model
    
    predictions = model.predict(X_test)
    print("RF Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    
    #Plotting of the model
    
    graph_name1= 'Random Forest'+'_without normalization w/o Opt'
    graph_name2=  'Random Forest'
    
    graph_1=folder+'Random Forest'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2=folder+'Random Forest'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=200)
        
        
        ConfusionMatrixDisplay.from_predictions(y_test, predictions,normalize=normalize,cmap=plt.cm.Reds)
        plt.title(title, size = 12)
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        
    savemodel= folder+ 'Random Forest'+'_model'+'.sav'
    joblib.dump(model, savemodel)
    
    
    Title1= folder+'RF'+'_Roc'+'.png'
    Title2= folder+'RF'+'_Precision_Recall'+'.png'
    plot_roc(model,Featurespace,classspace,classes,Title1,Title2)
