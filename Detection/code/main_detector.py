import numpy as np
import irlb
import argparse
import pickle as pk
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.special import softmax
from func_py2_postprocess import process_features

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--out_data', default="UCF10", type=str, help='Name of the Out-of-distribution dataset contained in the features folder')
parser.add_argument('--path', default="../features/resnet_10classfineIN10OUT", type=str, help='path to extracted features')


def main():
    global args
    args = parser.parse_args()
    print args.path

    process_features(args.path,args.out_data)


if __name__ == '__main__':
    main()

