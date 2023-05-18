# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:41:18 2023

@author: srpv
"""
import torch
from Trainer.pytorchtools import EarlyStopping
import torch.nn as nn
from blitz.modules import BayesianLinear,BayesianConv1d
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


class InterIntra_Train(torch.nn.Module):
    
  def __init__(self, backbone, feature_size=64, nb_class=3):
    super(InterIntra_Train, self).__init__()
    
    self.backbone = backbone

    self.Inter_sample_relation = torch.nn.Sequential(
                             BayesianLinear(feature_size*2, 256),
                             torch.nn.BatchNorm1d(256),
                             torch.nn.LeakyReLU(),
                             BayesianLinear(256, 1))
    
    
    self.Intra_temporal_head = torch.nn.Sequential(
        BayesianLinear(feature_size*2, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.LeakyReLU(),
        BayesianLinear(256, nb_class),
        torch.nn.Softmax())
    
    
  def aggregate(self, features, K):
    relation_pairs_list = list()
    targets_list = list()
    size = int(features.shape[0] / K)
    
    shifts_counter=1
    
    for index_1 in range(0, size*K, size):
        
      for index_2 in range(index_1+size, size*K, size):
        
        pos1 = features[index_1:index_1 + size]
        pos2 = features[index_2:index_2+size]
        pos_pair = torch.cat([pos1,
                              pos2], 1)  # (batch_size, fz*2)

        # Shuffle without collisions by rolling the mini-batch (negatives)
        neg1 = torch.roll(features[index_2:index_2 + size],
                          shifts=shifts_counter, dims=0)
        
        neg_pair1 = torch.cat([pos1, neg1], 1) # (batch_size, fz*2)

        relation_pairs_list.append(pos_pair)
        relation_pairs_list.append(neg_pair1)

        targets_list.append(torch.ones(size, dtype=torch.float32).cuda())
        targets_list.append(torch.zeros(size, dtype=torch.float32).cuda())
        
        shifts_counter+=1
        
        if(shifts_counter>=size):
            shifts_counter=1 # avoid identity pairs
    relation_pairs = torch.cat(relation_pairs_list, 0).cuda()  # K(K-1) * (batch_size, fz*2)
    targets = torch.cat(targets_list, 0).cuda()
    
    return relation_pairs, targets


  def run_test(self, predict, labels):
      correct = 0
      pred = predict.data.max(1)[1]
      correct = pred.eq(labels.data).cpu().sum()
      return correct, len(labels.data)


  def train(self, graph_name,tot_epochs, train_loader, opt):
      
    Training_loss=[]
    Training_accuracy=[]
    learn_rate=[]
    
    num_steps_per_epoch = math.floor(len(train_loader.dataset) / opt.batch_size)
    print("Num_steps_per_epoch....",num_steps_per_epoch)
    
    
    patience = opt.patience
    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
                  {'params': self.Inter_sample_relation.parameters()},
        {'params': self.Intra_temporal_head.parameters()},
    ], lr=0.01)
    
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * tot_epochs, eta_min=1e-4)
    
    BCE = torch.nn.BCEWithLogitsLoss()
    c_criterion = nn.CrossEntropyLoss()

    self.backbone.train()
    self.Inter_sample_relation.train()
    self.Intra_temporal_head.train()
    
    epoch_max = 0
    acc_max=0
    
    bayes_loop=opt.bayesian_train_size
    folder_created = os.path.join('Figures/', graph_name)
    print(folder_created)
    
    try:
        os.makedirs(folder_created, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
        
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(folder_created))
    
    for epoch in range(tot_epochs):

      acc_epoch=0
      acc_epoch_cls=0
      loss_epoch=0
      # the real target is discarded (unsupervised)
      for i, (data, data_augmented0, data_augmented1, data_label, _) in enumerate(train_loader):
          
        scheduler.step()  
        K = len(data) # tot augmentations
        x = torch.cat(data, 0).cuda()
        x_cut0 = torch.cat(data_augmented0, 0).cuda()
        x_cut1 = torch.cat(data_augmented1, 0).cuda()
        c_label = torch.cat(data_label, 0).cuda()
        
        #bayes_loop
        for _ in range(bayes_loop):
            
            optimizer.zero_grad()
            features = self.backbone(x)
            features_cut0 = self.backbone(x_cut0)
            features_cut1 = self.backbone(x_cut1)
            features_cls = torch.cat([features_cut0, features_cut1], 1)
            
            # aggregation function
            relation_pairs, targets = self.aggregate(features, K)
            
            # forward pass (relation head)
            score = self.Inter_sample_relation(relation_pairs).squeeze()
            c_output = self.Intra_temporal_head(features_cls)
            correct_cls, length_cls = self.run_test(c_output, c_label)
    
            # cross-entropy loss and backward
            loss = BCE(score, targets)
            loss_c = c_criterion(c_output, c_label)
            loss+=loss_c
    
            loss.backward()
            optimizer.step()
            
            # estimate the accuracy
            predicted = torch.round(torch.sigmoid(score))
            correct = predicted.eq(targets.view_as(predicted)).sum()
            accuracy = (100.0 * correct / float(len(targets)))
            
            accuracy_cls = 100. * correct_cls / length_cls
            
        lr_rate = scheduler.get_last_lr()[0]
        acc_epoch += accuracy.item()
        loss_epoch += loss.item()

        
        acc_epoch_cls += accuracy_cls.item()

      
      #len(train_loader)--> Dataset/ batch
      
      acc_epoch /= len(train_loader)
      Training_accuracy.append(acc_epoch)
      
      acc_epoch_cls /= len(train_loader)
      
      loss_epoch /= len(train_loader)
      Training_loss.append(loss_epoch)
      
      learn_rate.append(lr_rate)
      
      if (acc_epoch+acc_epoch_cls)>acc_max:
          acc_max = (acc_epoch+acc_epoch_cls)
          epoch_max = epoch

      early_stopping((acc_epoch+acc_epoch_cls), self.backbone)
      
      
      if early_stopping.early_stop:
          print("Early stopping")
          break

      if (epoch+1)%tot_epochs==0:
        print("[INFO] save backbone at epoch {}!".format(epoch))
        torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(folder_created,tot_epochs))

      print('Epoch [{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
            'Max ACC.= {:.1f}%, Max Epoch={},learning_rate={}' \
            .format(epoch + 1, 'Selftrain',
                    loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max,lr_rate))
          
          
    return Training_accuracy,Training_loss,learn_rate

