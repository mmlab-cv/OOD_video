import numpy as np
import irlb
import pickle as pk
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.special import softmax
from func_py2_postmax import process_features

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

## WRN C10
exp_dir = '../features/resnet_10class_Skihorse_noorto'

print exp_dir

out_data = 'UCFSkiHorse'
process_features(exp_dir,out_data)