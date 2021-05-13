# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

import os


global holdout
global holdout_train
global proc_img
holdout = 0 #9800
holdout_train = 0 #49800
proc_img = 0


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(directory + " was created")

def testData(net1, criterion, CUDA_DEVICE, trainloader10, testloader10, testloader, pathSave, dataName, extract_in, tune_param):

    savepath = "../features/" + pathSave + "/"
    

    ensure_dir(savepath)
    t0 = time.time()
    N = 1000
    MC_runs = 1
    if extract_in:
        print("Processing in-distribution images")
        ########################################In-distribution###########################################
        if not tune_param or T == 0:
            print("Processing in-distribution images: train")
            features = []
            labels = []
            N_train = 50000

            net1.eval()
            for j, data in enumerate(trainloader10):
                print(data)
                if j<holdout_train: continue
                images, lab = data

                inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
                labels_t = Variable(lab.cuda(CUDA_DEVICE))
                outputs, feat, weights = net1(inputs)
                #print(outputs)
                # print(torch.argmax(outputs).item(), lab.item())
                
                features.extend(list(feat.data.cpu().numpy()))
                labels.extend(labels_t.data.cpu())
                output.extend(list(outputs.data.cpu().numpy()))

                
                if j % 100 == 99:
                    print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(trainloader10), time.time()-t0))
                    t0 = time.time()
                
                if j == N_train - 1: break

            print(np.array(features).shape)
            np.save(savepath+'featuresTrain_in', np.array(features))
            np.save(savepath+'outputs_in',np.array(output))
            print(len(labels))
            np.save(savepath+'labelsTrain_in', labels)
            print(weights.shape)
            np.save(savepath+'weights', weights.data.cpu().numpy())
            if hasattr(net1.fc, 'bias'):
                if net1.fc.bias is not None:
                    np.save(savepath + 'bias', net1.fc.bias.cpu().detach().numpy())
            t0 = time.time()
        print("Processing in-distribution images: test")
        hold=0
        for j, data in enumerate(testloader10):
            if j<holdout:
                hold = proc_img
                continue
            if not tune_param:
                if j<proc_img: continue
            else:
                if j>=proc_img: break
                hold = 0
            
            images, lab = data

            net1.eval()
            with torch.no_grad():
                inputs = Variable(images.cuda(CUDA_DEVICE))
                outputs, feat, weights = net1(inputs)
                
                feat = feat[:, :, None]  
                for mc in range(1, MC_runs):
                    feat = torch.cat((feat, net1(inputs)[1][:, : , None]), dim=2)
                if j == hold:
                    features = feat.cpu()
                else:
                    features = torch.cat((features, feat.cpu()), dim=0)

                # print(features[-1, np.random.randint(0, 640), :])

                
            if j % 50 == 0:
                print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader10), time.time()-t0))
                t0 = time.time()
                
            
            if j == N - 1: break

        features = features.numpy()
        print(features.shape)
        np.save(savepath+'featuresTest_in', features)
        t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    features = []
    hold=0
    for j, data in enumerate(testloader):
        if j<holdout:
            hold = proc_img
            continue
        if not tune_param:
            if j<proc_img: continue
        else:
            if j>=proc_img: break
            hold = 0

        images, _ = data

        net1.eval()
        with torch.no_grad():
            inputs = Variable(images.cuda(CUDA_DEVICE))
            outputs, feat, weights = net1(inputs)
            feat = feat[:, :, None]
            for mc in range(1, MC_runs):
                feat = torch.cat((feat, net1(inputs)[1][:, :, None]), dim=2)
            if j == hold:
                features = feat.cpu()
            else:
                features = torch.cat((features, feat.cpu()), dim=0)


        
        if j % 50 == 0:
            print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader), time.time()-t0))
            t0 = time.time()
        
        if j== N-1: break

    features = features.numpy()
    print(features.shape)
    np.save(savepath+'features_out_'+dataName, features)




def testGaussian(net1, criterion, CUDA_DEVICE, trainloader10, testloader10, testloader, pathSave, dataName, extract_in, tune_param):
    savepath = "../features/" + pathSave + "/"

    ensure_dir(savepath)
    t0 = time.time()
    N = 10000
    MC_runs = 2

    if extract_in:
        print("Processing in-distribution images")
    ########################################In-distribution###########################################
        if not tune_param or T == 0:
            print("Processing in-distribution images: train")
            features = []
            labels = []
            N_train = 50000
            for j, data in enumerate(trainloader10):
                #if j<holdout_train: continue
                images, lab = data
                
                inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
                labels_t = Variable(lab.cuda(CUDA_DEVICE), requires_grad = True)

                outputs, feat, weights = net1(inputs)

                features.extend(list(feat.data.cpu().numpy()))
                labels.extend(labels_t.data.cpu())


                if j % 100 == 99:
                    print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(trainloader10), time.time()-t0))
                    t0 = time.time()
                
                if j == N_train - 1: break

            print(np.array(features).shape)
            np.save(savepath+'featuresTrain_in', np.array(features))
            print(len(labels))
            np.save(savepath+'labelsTrain_in', labels)
            print(weights.shape)
            np.save(savepath+'weights', weights.data.cpu().numpy())
            t0 = time.time()

        print("Processing in-distribution images: test")
        hold=0
        for j, data in enumerate(testloader10):
            if j<holdout:
                hold = proc_img
                continue
            if not tune_param:
                if j<proc_img: continue
            else:
                if j>=proc_img: break
                hold = 0
            images, lab = data

            net1.eval()
            with torch.no_grad():
                inputs = Variable(images.cuda(CUDA_DEVICE))
                outputs, feat, weights = net1(inputs)
                feat = feat[:, :, None]
                for mc in range(1, MC_runs):
                    feat = torch.cat((feat, net1(inputs)[1][:, : , None]), dim=2)
                if j == hold:
                    features = feat
                else:
                    features = torch.cat((features, feat), dim=0)

                # print(features[-1, np.random.randint(0, 640), :])


            if j % 50 == 0:
                print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader10), time.time()-t0))
                t0 = time.time()
            
            if j == N - 1: break

        features = features.cpu().numpy()
        print(features.shape)
        np.save(savepath+'featuresTest_in', features)
        t0 = time.time()
    ########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    features = []
    hold = 0
    for j, data in enumerate(testloader):
        if j<holdout:
            hold = proc_img
            continue
        if not tune_param:
            if j<proc_img: continue
        else:
            if j>=proc_img: break
            hold = 0
        
        images = torch.randn(1,3,32,32) + 0.5
        images = torch.clamp(images, 0, 1)
        
        images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
        images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
        images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
        

        net1.eval()
        with torch.no_grad():
            inputs = Variable(images.cuda(CUDA_DEVICE))
            outputs, feat, weights = net1(inputs)
            feat = feat[:, :, None]
            for mc in range(1, MC_runs):
                feat = torch.cat((feat, net1(inputs)[1][:, :, None]), dim=2)
            if j == hold:
                features = feat
            else:
                features = torch.cat((features, feat), dim=0)



        if j % 50 == 0:
            print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader), time.time()-t0))
            t0 = time.time()

        if j== N-1: break

    features = features.cpu().numpy()
    print(features.shape)
    np.save(savepath+'features_out_'+dataName,features)



def testUni(net1, criterion, CUDA_DEVICE, trainloader10, testloader10, testloader, pathSave, dataName, extract_in, tune_param):
    savepath = "../features/" + pathSave + "/"

    ensure_dir(savepath)
    t0 = time.time()
    N = 10000
    MC_runs = 50

    if extract_in:
        print("Processing in-distribution images")
    ########################################In-distribution###########################################
        if not tune_param or T == 0:
            print("Processing in-distribution images: train")
            features = []
            labels = []
            N_train = 50000
            for j, data in enumerate(trainloader10):
                #if j<holdout_train: continue
                images, lab = data
                
                inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
                labels_t = Variable(lab.cuda(CUDA_DEVICE),requires_grad = True)

                outputs, feat, weights = net1(inputs)

                features.extend(list(feat.data.cpu().numpy()))
                labels.extend(labels_t.data.cpu())


                if j % 100 == 99:
                    print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(trainloader10), time.time()-t0))
                    t0 = time.time()
                
                if j == N_train - 1: break

            print(np.array(features).shape)
            np.save(savepath+'featuresTrain_in', np.array(features))
            print(len(labels))
            np.save(savepath+'labelsTrain_in', labels)
            print(weights.shape)
            np.save(savepath+'weights', weights.data.cpu().numpy())
            t0 = time.time()

        print("Processing in-distribution images: test")
        hold=0
        for j, data in enumerate(testloader10):
            if j<holdout:
                hold = proc_img
                continue
            if not tune_param:
                if j<proc_img: continue
            else:
                if j>=proc_img: break
                hold = 0
            images, lab = data

            net1.eval()
            with torch.no_grad():
                inputs = Variable(images.cuda(CUDA_DEVICE))
                outputs, feat, weights = net1(inputs)
                feat = feat[:, :, None]
                for mc in range(1, MC_runs):
                    feat = torch.cat((feat, net1(inputs)[1][:, : , None]), dim=2)
                if j == hold:
                    features = feat
                else:
                    features = torch.cat((features, feat), dim=0)

                # print(features[-1, np.random.randint(0, 640), :])


            if j % 50 == 0:
                print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader10), time.time()-t0))
                t0 = time.time()
            
            if j == N - 1: break

        features = features.cpu().numpy()
        print(features.shape)
        np.save(savepath+'featuresTest_in', features)
        t0 = time.time()
    ########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    features = []
    hold=0
    for j, data in enumerate(testloader):
        if j<holdout:
            hold = proc_img
            continue
        if not tune_param:
            if j<proc_img: continue
        else:
            if j>=proc_img: break
            hold = 0
        
        images = torch.rand(1,3,32,32)
        images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
        images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
        images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)

        net1.eval()
        with torch.no_grad():
            inputs = Variable(images.cuda(CUDA_DEVICE))
            outputs, feat, weights = net1(inputs)
            feat = feat[:, :, None]
            for mc in range(1, MC_runs):
                feat = torch.cat((feat, net1(inputs)[1][:, :, None]), dim=2)
            if j == hold:
                features = feat
            else:
                features = torch.cat((features, feat), dim=0)



        if j % 50 == 0:
            print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader), time.time()-t0))
            t0 = time.time()

        if j== N-1: break

    features = features.cpu().numpy()
    print(features.shape)
    np.save(savepath+'features_out_'+dataName,features)
