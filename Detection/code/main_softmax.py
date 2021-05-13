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
import argparse
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
#import lmdb
from scipy import misc
#import caldrop as c
import calsoftmax as c

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--nn', default="3Dresnet", type=str,
                    help='neural network name and training set')
parser.add_argument('--path', default="resnet.py", type=str,
                    help='path to model')
parser.add_argument('--no_in_dataset', dest='in_dataset', action='store_false',
                    help='do not extract features for in-distribution dataset - Default=True')
parser.add_argument('--out_dataset', default="Imagenet_crop", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--gpu', default = 0, type = int,
                    help='gpu index')
parser.add_argument('--tune_param', action='store_true')
parser.set_defaults(argument=True)
parser.set_defaults(in_dataset=True)
parser.set_defaults(tune_param=False)




# Setting the name of neural networks

# Wide-ResNet trained on CIFAR-10:    wideresnet10
# Wide-ResNet trained on CIFAR-100:   wideresnet100
#nnName = "densenet10"

# Setting the name of the out-of-distribution dataset

# Tiny-ImageNet (crop):     Imagenet
# Tiny-ImageNet (resize):   Imagenet_resize
# LSUN (crop):              LSUN
# LSUN (resize):            LSUN_resize
# iSUN:                     iSUN
# Gaussian noise:           Gaussian
# Uniform  noise:           Uniform
#dataName = "Imagenet"

def main():
    global args
    args = parser.parse_args()
    c.test(args.nn, args.in_dataset, args.out_dataset, args.path, args.tune_param, args.gpu)

if __name__ == '__main__':
    main()

















