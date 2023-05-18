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
import numpy as np
from sklearn.svm import SVC
from Classifiers.Helper import *
from Classifiers.plot_roc import *
#%%

def SVM(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder):
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    random_state = np.random.RandomState(0)
    #svc_model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',decision_function_shape='ovo', verbose=True,random_state=None)
    svc_model = SVC(kernel='rbf', probability=True, random_state=random_state)
    model=svc_model.fit(X_train,y_train)
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model,X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))  
    predictions = model.predict(X_test)
    print("SVM Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    graph_name1= 'SVM'+'_without normalization w/o Opt'
    graph_name2=  'SVM'
    
    graph_1= folder+'SVM'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= folder+'SVM'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=200)
        
        
        ConfusionMatrixDisplay.from_predictions(y_test, predictions,normalize=normalize,cmap=plt.cm.Blues)
        plt.title(title, size = 12)
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        
        #disp.ax_.set_title(title)
        plt.title(title, size = 12)
        
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        
    savemodel=  folder+'SVM'+'_model'+'.sav'
    joblib.dump(model, savemodel)
    
    
    #plt.savefig('SVM_Training.png',bbox_inches='tight',dpi=800)
    #plt.show()
    
    
    Title1= folder+'SVM'+'_Roc'+'.png'
    Title2= folder+'SVM'+'_Precision_Recall'+'.png'
    plot_roc(model,Featurespace,classspace,classes,Title1,Title2)
    
    
    
    # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
    # plot_learning_curve(model, title, X_train, y_train, ylim=(0.2, 1.01),
    #                     cv=cv, n_jobs=4)

