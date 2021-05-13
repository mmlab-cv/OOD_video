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
from util.temporal_transforms import LoopPadding, TemporalRandomCrop
from torch.autograd import Variable
from util.target_transforms import ClassLabel, VideoID
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
CUDA_DEVICE = 0
from collections import OrderedDict

from util.spatial_transforms import (
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


criterion = nn.CrossEntropyLoss()


def test(nnName, args):

    import models.resnet as rsn


    d_rate = 0.0
    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    #3Dresnet
    if nnName == "3Dresnet":
        net1 = rsn.resnet34(num_classes = args.n_classes,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
        checkpoint = torch.load(args.trained_model_path)
        state_dict =checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        net1.load_state_dict(new_state_dict)
    
        optimizer1 = optim.SGD(net1.parameters(), lr = 0, momentum = 0)
        net1.cuda(CUDA_DEVICE)
        video_path_in = os.path.join(args.root_path, args.video_path_in)
        video_path_out = os.path.join(args.root_path, args.video_path_out)
        temporal_transform = LoopPadding(args.sample_duration)
        target_transform = ClassLabel()
        args.annotation_path_in = os.path.join(args.root_path, args.annotation_path_in)
        spatial_transform = Compose([
            Scale(args.sample_size),
            CenterCrop(args.sample_size),
            ToTensor(args.norm_value), norm_method
        ])
        validation_data = get_validation_set(args, args.annotation_path_in, video_path_in, spatial_transform, temporal_transform, target_transform)
        testloaderIn= torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads, pin_memory=True)
        training_data = get_treining_set(args, args.annotation_path_in, video_path_in, spatial_transform, temporal_transform, target_transform)
        trainloaderIn = torch.utils.data.DataLoader(
            training_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)
        
        temporal_transform = LoopPadding(args.sample_duration)
        target_transform = ClassLabel()
        args.annotation_path_out = os.path.join(args.root_path, args.annotation_path_out)
        spatial_transform = Compose([
            Scale(args.sample_size),
            CenterCrop(args.sample_size),
            ToTensor(args.norm_value), norm_method
        ])
        validation_OUTdata = get_validation_set(args, args.annotation_path_out, video_path_out, spatial_transform, temporal_transform, target_transform)
        testloaderOut= torch.utils.data.DataLoader(validation_OUTdata, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads, pin_memory=True)
        d.testData(net1, criterion, CUDA_DEVICE, trainloaderIn, testloaderIn, testloaderOut, args.path, args.in_dataset, args.tune_param)
