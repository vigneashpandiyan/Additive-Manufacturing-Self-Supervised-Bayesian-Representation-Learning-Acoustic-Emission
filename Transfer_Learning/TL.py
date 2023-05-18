# -*- coding: utf-8 -*-
import numpy as np
import torch
import Dataloader.transforms as transforms
from Dataloader.AE_dataset_loader import load_LPBF
from Dataloader.LPBF_loader_eval import LPBF_loader_eval
import torch.utils.data as data
from Trainer.pytorchtools import EarlyStopping
from Model.Network import ConvBayes,LinearBayes
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from blitz.modules import BayesianLinear,BayesianConv1d
from matplotlib import colors
import torch.nn.functional as F
import os


def transfer_learning(dataset,path,ckpt,lkpt,opt):
    
    
    dataset_name=str(dataset)+'_rawspace_5000.npy'
    dataset_label= str(dataset)+'_classspace_5000.npy'
    x_train, y_train, x_val, y_val, x_test, y_test, nb_class = load_LPBF(path,dataset_name,dataset_label)
    
    acc_test, epoch_max_point = transferlearning(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt,lkpt, opt,dataset)
    
    return acc_test, epoch_max_point

def transferlearning(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt,lkpt, opt,dataset):
    # no augmentations used for linear evaluation
    
    graph_name='Transfer_learning_'+str(dataset)
    
    folder_created = os.path.join('Figures/', graph_name)
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
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
    # if ckpt_tosave:
    #     torch.save(backbone_lineval.state_dict(), ckpt_tosave)
    print(opt.feature_size)
    linear_layer = LinearBayes(opt.feature_size,len(nb_class)).cuda()
    checkpoint = torch.load(lkpt, map_location='cpu')
    linear_layer.load_state_dict(checkpoint)
    

    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(linear_layer.parameters(), lr=opt.learning_rate_test)
    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience_test, verbose=True)
    best_acc = 0
    best_epoch = 0
    num=opt.bayesian_train_size
    print('Linear evaluation')
    for epoch in range(opt.epochs_test):
        linear_layer.train()
        backbone_lineval.eval()

        acc_trains = list()
        for i, (data, target) in enumerate(train_loader_lineval):
            
            for x in range(num):
                optimizer.zero_grad()
                data = data.cuda()
                target = target.cuda()
    
                output = backbone_lineval(data).detach()
                
                # print("eval_shape:",output.shape )
                
                output = linear_layer(output)
                loss = CE(output, target)
                loss.backward()
                optimizer.step()
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_trains.append(accuracy.item())

        print('[Train-{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
              .format(epoch + 1, 'Self Test_Bayesian', loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        linear_layer.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader_lineval):
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, None)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    torch.save(linear_layer.state_dict(),'{}/{}_linear.tar'.format(folder_created,graph_name))
        
        
    linear_layer.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for data, target in train_loader_lineval:
            
            
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data).detach()
            output = linear_layer(output)
            # estimate the accuracy
            prediction = output.argmax(-1)
            prediction=prediction.data.cpu().numpy()
            output=target.data.cpu().numpy()
            
            y_true.extend(output) # Save Truth 
            y_pred.extend(prediction) # Save Prediction
            
        plot_confusion_matrix(y_true, y_pred,folder_created,graph_name)

    return test_acc, best_epoch


            
   

#%%

def plot_confusion_matrix(y_true, y_pred,folder_created,graph_name):
    
    classes = ('1', '2', '3')
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float')  / cm.sum(axis=1)[:, np.newaxis]
    cmn=cmn*100
    
    fig, ax = plt.subplots(figsize=(12,9))
    sns.set(font_scale=3) 
    b=sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes,cmap="coolwarm",linewidths=0.1,annot_kws={"size": 25},cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts: b.set_text(b.get_text() + " %")
    plt.ylabel('Actual',fontsize=25)
    plt.xlabel('Predicted',fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize= 20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center",fontsize= 20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname=folder_created+'/'+str(graph_name)+'_Classification_accuracy.png'
    plotname=str(plotname)
    plt.savefig(plotname,bbox_inches='tight')
    plt.show()
    plt.close()

