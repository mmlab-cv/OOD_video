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
import cal as c

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--nn', default="3Dresnet", type=str, help='neural network name and training set')
parser.add_argument('--path', default="resnet.py", type=str, help='path to model')
parser.add_argument('--no_in_dataset', dest='in_dataset', action='store_false', help='do not extract features for in-distribution dataset - Default=True')
parser.add_argument('--out_dataset', default="Imagenet_crop", type=str, help='out-of-distribution dataset')
parser.add_argument('--gpu', default = 0, type = int, help='gpu index')
parser.add_argument('--tune_param', action='store_true')
parser.set_defaults(argument=True)
parser.set_defaults(in_dataset=True)
parser.set_defaults(tune_param=False)

parser.add_argument('--n_classes', default=8, type=int, help='Number of classes ( ucf101: 101, hmdb51: 51)' )
parser.add_argument( '--sample_size', default=112, type=int, help='Height and width of inputs')
parser.add_argument( '--sample_duration', default=16,type=int,  help='Temporal duration of inputs')
parser.add_argument( '--norm_value', default=1,  type=int,  help= 'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
parser.add_argument( '--scales', default=1, type=int, help='Number of scales for multiscale cropping')
parser.add_argument( '--dataset',default='ucf101',  type=str,   help='Used dataset (olympicSports | ucf101 | hmdb51)')
parser.add_argument( '--batch_size', default=16, type=int, help='Batch Size')
parser.add_argument( '--n_val_samples', default=3, type=int,help='Number of validation samples for each activity')
parser.add_argument( '--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
parser.add_argument( '--root_path', default='/root/data/UFC101', type=str, help='Root directory path of data')
parser.add_argument( '--video_path_in', default='video_ucf101_jpg', type=str, help='Directory path of Videos in in-distribution dataset')
parser.add_argument( '--video_path_out', default='video_ucf101_jpg', type=str, help='Directory path of Videos out-of-distribution dataset')
parser.add_argument( '--annotation_path_out', default='ucf101_in.json', type=str, help='Annotation file path out of out-of-distribution dataset')
parser.add_argument( '--annotation_path_in', default='ucf101_out.json', type=str, help='Annotation file path in in-distribution dataset')
parser.add_argument( '--trained_model_path', default='olympicSports/results/save_200.pth', type=str, help='Path to the trained model')
#parser.add_argument('--mode', default="freeze", type=str, help='mode of the network' )


def main():
    global args
    args = parser.parse_args()
    c.test(args.nn, args)

if __name__ == '__main__':
    main()

















