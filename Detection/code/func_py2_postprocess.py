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


debug = False

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

def calculate_ROM(A):
    USV = irlb.irlb(A, 101)
    first_sing_vec = USV[0][:, 0]
    first_sing_val = USV[1][0]
    norm = np.linalg.norm(A, ord='fro')
    ROM = first_sing_val / norm
    USVV = USV[1]/norm

    if debug:
        print ROM, USV[1]/norm
    return ROM, first_sing_vec, USVV


def preprocess(D, labels=None):
    if labels is None:
        data = np.array(D)
        D_out = data.transpose()
    else:
        D_out = []
        for l in set(labels):
            data = np.array(D[labels == l])
            if len(data) != 0:
                D_out.append(data.transpose())
    return D_out


def correlation(A, B):
    corr = np.matmul(A, B)
    if len(B.shape) == 2:
        corr /= np.linalg.norm(B, axis=0) + 1e-4
    elif len(B.shape) == 3:
        corr /= np.linalg.norm(B, axis=1)[:, None, :] + 1e-8
    corr = np.abs(corr)
    return corr


def classify(D, score_func, weights, first_sing_vecs):
    score = score_func(D, weights, first_sing_vecs)
    if debug:
        print np.percentile(score, q=10),
        print np.percentile(score, q=50),
        print np.percentile(score, q=90)

    return score

def score_func(D, weights, first_sing_vecs):
    measure = first_sing_vecs
    corr = correlation(measure, D)
    score = np.arccos(corr)
    if len(corr.shape) == 3:
        score = np.min(score, axis=1)
        score = np.min(score, axis=0)

    elif len(corr.shape) == 2:
        score = np.min(score, axis=0)

    return score

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

    print test.shape
    print test_in.shape
    print train.shape

    labels = np.squeeze(np.load(exp_dir + '/labelsTrain_in.npy'))
    
    weights = np.load(exp_dir + '/weights.npy').transpose()


    print  np.dot(np.transpose(weights),weights)
    bias = []
    try:
        bias = np.load(exp_dir + '/bias.npy')
    except:
        print("no bias")
        print len(test_in)
    train = preprocess(train, labels)
    test = preprocess(test)
    test_in = preprocess(test_in)
    first_sing_vecs = []
    ROMs = []
    USVs = []


    
    for l, data in enumerate(train):
        print data.shape
        if len(data) != 0:
            ROM, u, USV = calculate_ROM(data)
            first_sing_vecs.append(u)
            ROMs.append(ROM)
            USVs.append(USV)
    first_sing_vecs = np.array(first_sing_vecs)
    
    #np.save('firstvec', first_sing_vecs)
    
    S_U = np.matmul(first_sing_vecs, np.transpose(first_sing_vecs))
    S_W = np.matmul(np.transpose(weights), weights)
    S_U_V = np.matmul(first_sing_vecs, weights)
    if debug:
        print S_U
        print S_W
        print S_U_V
    
    if debug:
        print '+++++++++++++++'

    score_in = classify(test_in, score_func, weights, first_sing_vecs)
    target_in = np.zeros_like(score_in)


    score_out = classify(test, score_func, weights, first_sing_vecs)
    target_out = np.ones_like(score_out)

    

    targets = np.concatenate((target_in, target_out))
    scores = np.concatenate((score_in, score_out))
    new_train = []
    new_vec = np.zeros((512,216))
    
    maximo = 10000
    for i in train:
        for j in i:
            if len(j) < maximo:
                maximo = len(j)
    print maximo

    for i in train:
        nuovo_vet = copy.deepcopy(new_vec)
        nuovo_vet = i[:,:209]
        new_train.append(nuovo_vet)

    data_dic = {}
    data_dic['score_in'] = score_in
    data_dic['score_out'] = score_out
    data_dic['weights'] = weights
    data_dic['sing_vecs'] = first_sing_vecs
    data_dic['ROM'] = ROMs
    data_dic['train'] = new_train
    data_dic['test_out'] = test
    data_dic['test_in'] = test_in
    data_dic['bias'] = bias
    

    print 'scoresIN', scores[0:30]
    print 'scoresOUT', scores[-30:]

    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    fpr95 = fpr[tpr >= 0.95][0]
    tpr95 = tpr[tpr >= 0.95][0]
    print 'FPR @95TPR:', fpr95

    print 'Detection Error:', np.min(0.5 * (1 - tpr) + 0.5 * fpr)

    print 'AUC: ', metrics.auc(fpr, tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(1 - targets,1 -  scores)
    print 'AUPRin:', metrics.auc(recall, precision)
    
    np.seterr(divide='ignore', invalid='ignore')
    F1 = 2 * (precision * recall) / (precision + recall)
    print 'F1-scorein: ', np.max(F1)

    precision, recall, thresholds = metrics.precision_recall_curve(targets, scores)
    print "thresholds"
    count = 0
    for el in precision:
        if el != 0:
            count += 1
    print count
    print 'AUPRout:', metrics.auc(recall, precision)
    F1 = 2 * (precision * recall) / (precision + recall)
    print 'F1-scoreout: ', np.max(F1), '@', thresholds[np.argmax(F1)]*180/np.pi, ' degrees'

    #PLOT
    print len(USVs)
    print len(USVs[1])
    RESs = []
    for i in range(0,len(USVs[1])):
        res = 0
        for j in range(0,len(USVs)):
            res = res + USVs[j][i]
        res = res/len(USVs)
        #print res
        RESs.append(res)
    '''
    x = np.arange(0, 100, 1.)
    plt.plot(x, RESs, 'bo')
    plt.show
    '''
    data_dic['USVs'] = RESs
    sio.savemat(exp_dir + '/' + out_dataset, mdict=data_dic, appendmat=True)