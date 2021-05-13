import math
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def block_mean(ar, fact):
    sx = len(ar)
    #resurned = ar [3000:5000]
    resurned = ar[0:sx/fact:]
    return resurned

def block_mean_thereal(ar, fact):
    sx = len(ar)
    final = int(math.floor(sx/fact))
    print(final)
    resurned = []
    for i in range(final):
        resurned.append(math.floor(sum(ar[(i*fact):((1+i)*fact)])/fact))
    return resurned

def correlation(A, B):
    print A.shape
    print B.shape
    corr = np.matmul(A, B)
    print B.shape
    corr = np.abs(corr)
    if len(B.shape) == 2:
        corr /= np.linalg.norm(B, axis=0) + 1e-4
    elif len(B.shape) == 3:
        corr /= np.linalg.norm(B, axis=1)[:, None, :] + 1e-8
    
    return corr




matrice = loadmat('../features/hmdb25_25/HMDB25.mat')
matrice1 = loadmat('../features/resnet_10class_Skihorse_LSS201_drop/UCF10.mat')
matrice2 = loadmat('../features/resnet_50classbox_no_orto/UCFBox.mat')

RESs2 = matrice1['USVs']
RESs = matrice['USVs']
sing_vec = matrice['sing_vecs']
sing_vec2 = matrice2['sing_vecs']
feature_in = matrice['test_in']
feature_in_2 = matrice2['test_in']
feature_out = matrice['test_out']
feature_out_2 = matrice1['test_out']
feat_train = matrice['train']
weight = matrice['weights']
#print weight.shape
for i in range(0,10):
    for j in range(i+1,10):
        cirogiu = np.dot(weight[i][:],weight[j][:])
        #print cirogiu
print sing_vec.shape
for i in range(0,10):
	for j in range(i,10):
		giuseppi = np.dot(sing_vec[i][:],sing_vec[j][:])
		#print i, j, giuseppi

for i in range(0,10):
	for j in range(0,10):
		giuseppi = np.dot(sing_vec[i],feat_train[j])
		fin = sum(giuseppi)
		#print i, j, fin

'''
d = dict()
for i in feature_in:
	for s in i:
		i2 = tuple(s.tolist())
		
		if d.get(i2) == None:
			d[i2] = 0
		d[i2] += 1
#print d.values()

e = dict()
for i in feature_in:
	for s in i:
		i2 = tuple(s.tolist())
		
		if e.get(i2) == None:
			e[i2] = 0
		e[i2] += 1
#print e.values()
f = dict()
for i in feature_in:
	for s in i:
		i2 = tuple(s.tolist())
		
		if f.get(i2) == None:
			f[i2] = 0
		f[i2] += 1
#print f.values()
'''
#print len(feature_in)
#print len(feature_out)
print feat_train.shape
print sing_vec.shape





corr = correlation(sing_vec, feature_in)
phi = np.arccos(corr)
if len(corr.shape) == 3:
        phi = np.min(phi, axis=1)
        phi = np.min(phi, axis=0)*180

elif len(corr.shape) == 2:
    phi = np.min(phi, axis=0)*180

corr_ood = correlation(sing_vec, feature_out)
phi_ood = np.arccos(corr_ood)
if len(corr_ood.shape) == 3:
        phi_ood = np.min(phi_ood, axis=1)
        phi_ood = np.min(phi_ood, axis=0)*180

elif len(corr_ood.shape) == 2:
    phi_ood = np.min(phi_ood, axis=0)*180

corr_ood_2 = correlation(sing_vec, feature_out_2)
phi_ood_2 = np.arccos(corr_ood_2)
if len(corr_ood_2.shape) == 3:
        phi_ood_2 = np.min(phi_ood_2, axis=1)
        phi_ood_2 = np.min(phi_ood_2, axis=0)*180

elif len(corr_ood_2.shape) == 2:
    phi_ood_2 = np.min(phi_ood_2, axis=0)*180

corr_train = correlation(sing_vec, feat_train)
phi_train = np.arccos(corr_train)
if len(corr_train.shape) == 3:
        phi_train = np.min(phi_train, axis=1)
        phi_train = np.min(phi_train, axis=0)*180

elif len(corr_train.shape) == 2:
    phi_train = np.min(phi_train, axis=0)*180

#phi = block_mean_thereal(phi_ood,2)



#plt.hist(phi_train, bins = 25, color='g',density=False,histtype='step', label='Train')
plt.hist(phi, bins = 80, color='r',density=False,histtype='step', label='In Distribution')
plt.hist(phi_ood, bins = 80, color='b',density=False, histtype='step', label='Out of Distribution')
#plt.hist(phi_ood_2, bins = 35, color='g',density=False, histtype='step', label='Out of Distribution 2 classes')
plt.gca().set(ylabel='Count', xlabel='Spectral Discrepancy (degrees)')
plt.legend()
#plt.yscale("log")
#plt.xscale("log")
plt.show()



#PLOT ENERGIA

x = np.arange(1, 102, 1.)
y = np.arange(1, 102, 1.)

plt.plot(y, RESs[0], color = 'k',marker='v',  label='w proposed structure')
plt.plot(x, RESs2[0], color = 'b',marker='v',  label='w/o proposed structure')
plt.gca().set(ylabel='Energy Ratio', xlabel='Singular Vector Index')
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.show()
