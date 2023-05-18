
import matplotlib.pyplot as plt
import numpy as np

def training_plots(Training_loss,Training_accuracy,learning_rate):

    Training_loss=np.asarray(Training_loss)
    Training_loss=Training_loss.astype(np.float64)
    classfile = 'Training_loss'+'.npy'
    np.save(classfile,Training_loss, allow_pickle=True)
    
    
    Training_accuracy=np.asarray(Training_accuracy)
    Training_accuracy=Training_accuracy.astype(np.float64)
    classfile = 'Training_accuracy'+'.npy'
    np.save(classfile,Training_accuracy, allow_pickle=True)
    
    learning_rate=np.asarray(learning_rate)
    learning_rate=learning_rate.astype(np.float64)
    classfile = 'learning_rate'+'.npy'
    np.save(classfile,learning_rate, allow_pickle=True)
    
    
    fig = plt.figure(figsize = (5, 3))
    plt.plot(Training_loss,'blue') 
    plt.title('Training loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(labels=['Training loss'], loc='best',fontsize=15,frameon=False)
    plotname='Training_loss'+'.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=200)
    plt.show()
    
    fig = plt.figure(figsize = (5, 3))
    plt.plot(Training_accuracy,'red') 
    plt.title('Training accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(labels=['Training accuracy'], loc='best',fontsize=15,frameon=False)
    plotname='Training_accuracy'+'.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=200)
    plt.show()
    
    
    fig = plt.figure(figsize = (5, 3))
    plt.plot(learning_rate,'black') 
    plt.title('learning_rate')
    plt.xlabel("Epoch")
    plt.ylabel("learningrate")
    plt.legend(labels=['learning_rate'], loc='best',fontsize=15,frameon=False)
    plotname='learning_rate'+'.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=200)
    plt.show()