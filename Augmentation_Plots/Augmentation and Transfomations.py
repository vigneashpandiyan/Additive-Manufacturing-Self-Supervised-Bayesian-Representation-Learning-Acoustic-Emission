# -*- coding: utf-8 -*-
"""
Created on Sat May  6 01:06:28 2023

@author: srpv

['jitter','scaling','cutout','magnitude_warp','time_warp','window_slice','window_warp']

"""


import numpy as np
import augmentation as aug
import helper as hlp
import matplotlib.pyplot as plt

#%%


path=r'C:\Users\srpv\Desktop\LPBF Bayesian Self-Time learning\Data'
dataset_name='D1_rawspace_5000.npy'
dataset_label= 'D1_classspace_5000.npy'
x_train, y_train, x_val, y_val, x_test, y_test, nb_class= aug.load_LPBF(path, dataset_name,dataset_label)

length=x_train.shape[1]
t0=0
dt=1/400000
t = np.arange(0, length) * dt + t0  


hlp.plot1d(x_train[0],t,save_file='Original_signal.png')



# ## Jittering
jitter = aug.jitter(x_train)[0]
hlp.plot_d(x_train[0],t, jitter, save_file='Jittering.png',Transformation='Jittering')
# hlp.plot1d(x_train[0], aug.jitter(x_train)[0])

## Scaling
scaling = aug.scaling(x_train)[0]
hlp.plot_d(x_train[0],t, scaling, save_file='Scaling.png',Transformation='Scaling')
# hlp.plot1d(x_train[0], aug.scaling(x_train)[0])

## Rotation
rotation = aug.rotation(x_train)[0]
hlp.plot_d(x_train[0],t, rotation, save_file='Rotation.png',Transformation='Rotation')
# hlp.plot1d(x_train[0], aug.rotation(x_train)[0])

## Cutout
cutout = aug.cutout((x_train)[0])
hlp.plot_d(x_train[0],t, cutout, save_file='Cutout.png',Transformation='Cut out')
# hlp.plot1d(x_train[0], aug.cutout(x_train)[0])


## Permutation
permutation = aug.permutation(x_train)[0]
hlp.plot_d(x_train[0],t, permutation, save_file='Permutation.png',Transformation='Permutation')
# hlp.plot1d(x_train[0], aug.permutation(x_train)[0])


## Magnitude Warping
magnitude_warp = aug.magnitude_warp(x_train)[0]
hlp.plot_d(x_train[0],t, magnitude_warp, save_file='Magnitude Warping.png',Transformation='Magnitude warping')
# hlp.plot1d(x_train[0], aug.magnitude_warp(x_train)[0])

## Time Warping
time_warp = aug.time_warp(x_train)[0]
hlp.plot_d(x_train[0],t, time_warp, save_file='Time Warping.png',Transformation='Time warping')
# hlp.plot1d(x_train[0], aug.time_warp(x_train)[0])


## Window Slicing
window_slice = aug.window_slice(x_train)[0]
hlp.plot_d(x_train[0],t, window_slice, save_file='Window Slicing.png',Transformation='Window slicing')
# hlp.plot1d(x_train[0], aug.window_slice(x_train)[0])

## Window Warping
window_warp = aug.window_warp(x_train)[0]
hlp.plot_d(x_train[0],t, window_warp, save_file='Window Warping.png',Transformation='Window warping')
# hlp.plot1d(x_train[0], aug.window_warp(x_train)[0])

#%%
colour=['red', 'blue','green', 'pink','orange','purple','red','cyan','yellow','limegreen','black']
funct=['Original', 'Jitter', 'Scaling','Rotation', 'Cutout','Permutation','Magnitude_warp','Time_warp','Window_slice','Window_warp']

data_created=[(x_train)[0],jitter,scaling,rotation,cutout,permutation,magnitude_warp,time_warp,window_slice,window_warp]


def plot_time_series(x,t, x_, ax,colour,transformation):
    
    
   
   
    ax.plot(t,x.ravel(),'black', color='0.4',linewidth=2) 
    ax.plot(t,x_.ravel(),colour,ls='--',linewidth=0.75)  
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    ax.set_xlabel("time (ms)",fontsize=15)
    ax.set_ylabel('Amplitude',fontsize=15)
    ax.set_title(transformation,fontsize=15)
   
   
    ax.legend(['Original signal',str(transformation)],fontsize=12,bbox_to_anchor=(1.1, 1.05))
    
    ax.ticklabel_format(scilimits=(-3, 3))
    # ax.ticklabel_format(useMathText=True)
    
    ax.set_ylim([-1.2,1.2])


x=(x_train)[0]
fig, axs = plt.subplots(
    nrows=len(funct),
    ncols=1,
    sharey=False,
    figsize=(7, 18),
    dpi=800
  )

for i, key in enumerate(funct):
  print(i) 
  ax = axs.flat[i]
  y=data_created[i]
  transformation=(str(key)) 
  print(transformation)
  
  if i==0:
      ax.plot(t,y.ravel(),'black', color='0.4',linewidth=2) 
      ax.set_xlabel("time (ms)",fontsize=15)
      ax.set_ylabel('Amplitude',fontsize=15)
      ax.set_title(transformation,fontsize=15)
      ax.tick_params(axis='both', which='major', labelsize=12)
      ax.tick_params(axis='both', which='minor', labelsize=12)
      ax.legend(['Original signal'],fontsize=12,bbox_to_anchor=(1.1, 1.05))
      
  else:
      
      plot_time_series(x,t, y, ax,colour[i],transformation)
  
fig.tight_layout();
graphname='Augmentation'+'_'+'AE'+'signal'+'.png'
plt.savefig(graphname,bbox_inches='tight',pad_inches=0.1)
plt.show()