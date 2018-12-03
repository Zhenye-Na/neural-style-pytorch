"""
PyTorch implementation of paper "A Neural Algorithm of Artistic Style".

High level pipeline.

@author: Zhenye Na
"""

import os
import torch
import argparse


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--dataroot', type=str, default="../../../data", help='path to dataset')
    parser.add_argument('--ckptroot', type=str, default="../model/", help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0., help='beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, help='pre-trained epochs')
    parser.add_argument('--batch_size_train', type=int, default=128, help='training set input batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, help='test set input batch size')

    # training settings
    parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
    parser.add_argument('--cuda', type=bool, default=True, help='whether training using GPU cudatoolkit')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """High level pipeline for Neural Style Transfer."""
    pass


if __name__ == '__main__':
    main()
