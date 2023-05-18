# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:56:21 2023

@author: srpv
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_d(x,t, x_, save_file='',Transformation=''):
    fig = plt.figure(figsize = (9, 4))
    plt.title('Original signal vs ' +str(Transformation),fontsize=15)
   
    # min=np.min(x_.ravel())
    # max=np.max(x_.ravel())
    
    plt.plot(t,x.ravel(),'black', color='0.4',linewidth=2) 
    plt.plot(t,x_.ravel(),'blue',ls='--',linewidth=0.75)  
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.xlabel("time (ms)",fontsize=15)
    plt.ylabel('Amplitude',fontsize=15)
    
   
    plt.legend(['Original signal',str(Transformation)],fontsize=12)
    
    plt.ylim([-1.2,1.2])
    
    
    # plotname=str(Material)+'_'+str(classname)+'_rawsignal.png'
    plt.savefig(save_file,dpi=200)
    plt.show()



def plot1d(x,t,save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    
    plt.title('Original signal',fontsize=15)
    plt.plot(t, x,'black', color='0.4')
    
    
    plt.xlabel("time (ms)",fontsize=15)
    plt.ylabel('Amplitude',fontsize=15)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.legend(['Original signal'],fontsize=12)
    
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file,dpi=100)
        plt.show()
    