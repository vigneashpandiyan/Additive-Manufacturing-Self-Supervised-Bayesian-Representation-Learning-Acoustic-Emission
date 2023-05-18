# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:36:29 2023

@author: srpv
"""

import torch

from Model.Network import ConvBayes
from Trainer.Relational_Reasoning_Trainer import InterIntra_Train
from Dataloader.LPBF_InterIntra import LPBF_Inter_Intra_reasoning
import Dataloader.transforms as transforms
from Visualization.Training_plots import training_plots
import torch.utils.data as data
from prettytable import PrettyTable


import os

def Bayesian_backbone(x_train, y_train, opt,graph_name):
    
    
    folder_created = os.path.join('Figures/', graph_name)
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    
    K = opt.K #'Number of augmentation for each sample'
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    

    prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()
    
    if '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        nb_class=3
    else:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        nb_class=2
    

    backbone = ConvBayes(opt.feature_size).cuda()
    
    model = InterIntra_Train(backbone, feature_size, nb_class).cuda()

    train_set = LPBF_Inter_Intra_reasoning(data=x_train, targets=y_train, K=K,
                                        transform=train_transform, transform_cut=cut_piece,
                                        totensor_transform=tensor_transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(folder_created))
    
    Training_accuracy,Training_loss,learning_rate = model.train(graph_name,tot_epochs=tot_epochs, train_loader=train_loader, opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(folder_created))
    
    training_plots(Training_loss,Training_accuracy,learning_rate)

    return model

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
        
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params