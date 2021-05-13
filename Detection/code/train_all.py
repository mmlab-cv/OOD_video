import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


subprocess.run(f"CUDA_VISIBLE_DEVICES={args.gpu} python train_wideresnet.py --netmode freeze --dataset cifar10 --layers 28 --widen-factor 10 --name WideResNet-CIFAR10-freeze ", shell=True)
subprocess.run(f"CUDA_VISIBLE_DEVICES={args.gpu} python train_wideresnet.py --netmode freeze --dataset cifar100 --layers 28 --widen-factor 10 --name WideResNet-CIFAR100-freeze ", shell=True)
