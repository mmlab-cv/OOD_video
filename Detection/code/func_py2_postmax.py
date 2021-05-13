import numpy as np
import irlb
import pickle as pk
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.special import softmax
import scipy.io as sio
import random
import copy
import matplotlib

def process_features(exp_dir,out_dataset):

    train = np.squeeze(np.load(exp_dir + '/featuresTrain_in.npy'))
    '''
    print train.shape
    for i in range(0,50):
        for j in range(i,50):
            rus = np.matmul(train[i],train[j])
            print(i,j,rus)
    '''
    
    test = np.squeeze(np.load(exp_dir + '/features_out_'+out_dataset+'.npy'))
    test_in = np.squeeze(np.load(exp_dir + '/featuresTest_in.npy'))

    labels = np.squeeze(np.load(exp_dir + '/labelsTrain_in.npy'))
    
    weights = np.load(exp_dir + '/weights.npy').transpose()
    # global bias
    

    print test[0]