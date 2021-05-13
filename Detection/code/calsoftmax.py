# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""
from __future__ import print_function
import os
from dataset import get_training_set, get_validation_set, get_test_set, get_treining_set
import torch
from temporal_transforms import LoopPadding, TemporalRandomCrop
from torch.autograd import Variable
from target_transforms import ClassLabel, VideoID
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
#CUDA_DEVICE = 0
from collections import OrderedDict

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ToPILImage(),
    # transforms.RandomCrop(32),
    # transforms.ToTensor(),
    #transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])





# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nnName = "densenet10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()

class Opt:
    def __init__(self):
        self.num_classes = 10
        self.sample_size = 112
        #self.sample_duration = 16
        self.norm_value = 1
        self.scales = 1
        self.dataset = "ucf101"
        self.n_val_samples = 3
        self.batch_size = 8
        self.n_threads = 4



class Opt_50(Opt):
    def __init__(self):
        super().__init__()
        self.root_path = '../../../../../../media/mmlab/Volume/zenosambugaro/UCF100_dataset'
        self.video_path = os.path.join(self.root_path, "jpg_video")
        self.sample_duration = 16

class Opt_101(Opt):
    def __init__(self):
        super().__init__()
        self.root_path = '../../../../../../media/mmlab/Volume/zenosambugaro/UCF101_dataset'
        self.video_path = os.path.join(self.root_path, "jpg_video")
        self.sample_duration = 16





def test(nnName, extract_in_features, dataName, modelPath, tune_param, CUDA_DEVICE):
    
    #if netmode == 'base':
    #    import wideresnet_base as wrn
    #elif netmode == 'freeze':
    #    import wideresnet_freeze as wrn
    #import resnet as rsn
    import resnet as rsn

    opt_50 = Opt_50()
    opt_101 = Opt_101()
    d_rate = 0.0
    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    #3Dresnet
    if nnName == "3Dresnet":
        net1 = rsn.resnet34(num_classes = opt_50.num_classes,
                sample_size=opt_50.sample_size,
                sample_duration=opt_50.sample_duration)

        #DECOOMENTARE

        checkpoint = torch.load("../../../../../../media/mmlab/Volume/zenosambugaro/UCF100_dataset/results_10finetune/save_200.pth")
        state_dict =checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        net1.load_state_dict(new_state_dict)
    
        optimizer1 = optim.SGD(net1.parameters(), lr = 0, momentum = 0)
        net1.cuda(CUDA_DEVICE)
        
        temporal_transform = LoopPadding(opt_50.sample_duration)
        target_transform = ClassLabel()
        opt_50.annotation_path = os.path.join(opt_50.root_path, "ucfTrainTestlist_10/ucf101_01.json")
        spatial_transform = Compose([
            Scale(opt_50.sample_size),
            CenterCrop(opt_50.sample_size),
            ToTensor(opt_50.norm_value), norm_method
        ])
        validation_data = get_validation_set(opt_50, spatial_transform, temporal_transform, target_transform)
        testloaderIn= torch.utils.data.DataLoader(validation_data, batch_size=opt_50.batch_size, shuffle=False, num_workers=opt_50.n_threads, pin_memory=True)
        training_data = get_treining_set(opt_50, spatial_transform, temporal_transform, target_transform)
        trainloaderIn = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt_50.batch_size,
            shuffle=False,
            num_workers=opt_50.n_threads,
            pin_memory=True)
        
        temporal_transform = LoopPadding(opt_101.sample_duration)
        target_transform = ClassLabel()
        opt_101.annotation_path = os.path.join(opt_101.root_path, "ucfTrainTestlist_10class_1/ucf101_01.json")
        spatial_transform = Compose([
            Scale(opt_101.sample_size),
            CenterCrop(opt_101.sample_size),
            ToTensor(opt_101.norm_value), norm_method
        ])
        validation_OUTdata = get_validation_set(opt_101, spatial_transform, temporal_transform, target_transform)
        testloaderOut= torch.utils.data.DataLoader(validation_OUTdata, batch_size=opt_101.batch_size, shuffle=False, num_workers=opt_101.n_threads, pin_memory=True)
        d.testData(net1, criterion, CUDA_DEVICE, trainloaderIn, testloaderIn, testloaderOut, modelPath, dataName, extract_in_features, tune_param)
