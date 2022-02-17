import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils.lsun_loader as lsun_loader
import utils.svhn_loader as svhn
from utils.tinyimages_80mn_loader import RandomImages
from utils.imagenet_rc_loader import ImageNet

import pathlib

'''
This script makes the datasets used in training. The main function is make_datasets. 
'''


# *** update this before running on your machine ***
# paths = 'JBN'
paths = 'JKS'

if paths == 'JKS':

    cifar10_path = '../data/cifarpy'
    cifar100_path = '../data/cifar-100-python'
    svhn_path = '/nobackup-slow/dataset/svhn/'
    lsun_c_path = '/nobackup-slow/dataset/LSUN_C'
    lsun_r_path = '/nobackup-slow/dataset/LSUN_resize'
    isun_path = '/nobackup-slow/dataset/iSUN'
    dtd_path = '/nobackup-slow/dataset/dtd/images'
    places_path = '../data/places365/'
    tinyimages_300k_path = '/u/k/a/katzsamuels/ssnd_experiments/data/300K_random_images.npy'
    svhn_path = '/u/k/a/katzsamuels/ssnd_experiments/data/svhn'

elif paths == 'JBN':

    cifar10_path = '../data/cifarpy'
    cifar100_path = '../data/cifar-100-python'
    svhn_path = '/nobackup/svhn_jnakhleh'
    lsun_c_path = '/nobackup/lsun_c_jnakhleh/LSUN'
    lsun_r_path = '/nobackup/lsun_r_jnakhleh/LSUN_resize'
    isun_path = '/nobackup/isun_jnakhleh/iSUN'
    dtd_path = '/nobackup/dtd_jnakhleh/dtd/images'
    places_path ='/nobackup/places_jnakhleh/extracted/test_256'
    tinyimages_300k_path = '/nobackup/tinyimages_300k_jnakhleh/data/300K_random_images.npy'

# cifar10_path = '../data/cifarpy'
# cifar100_path = '../data/cifar-100-python'
# svhn_path = '../data/svhn/'
# lsun_c_path = '../data/LSUN_C'
# lsun_r_path = '../data/LSUN_resize'
# isun_path = '../data/iSUN'
# dtd_path = '../data/dtd/images'
# places_path = '../data/places365/'
# tinyimages_300k_path = '../data/300K_random_images.npy'
# svhn_path = '../data/svhn'



def load_tinyimages_300k(in_dset):
    print('loading TinyImages-300k')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    t = trn.Compose([trn.ToTensor(),
                     trn.ToPILImage(),
                     trn.ToTensor(),
                     trn.Normalize(mean, std)])

    data = RandomImages(root=tinyimages_300k_path, transform=t)

    return data


def load_CIFAR(dataset, classes=[]):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
    #                                trn.ToTensor(), trn.Normalize(mean, std)])
    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset in ['cifar10']:
        print('loading CIFAR-10')
        train_data = dset.CIFAR10(
            cifar10_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform, download=True)

    elif dataset in ['cifar100']:
        print('loading CIFAR-100')
        train_data = dset.CIFAR100(
            cifar100_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            cifar100_path, train=False, transform=test_transform, download=True)

    return train_data, test_data


def load_SVHN(include_extra=False):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    print('loading SVHN')
    if not include_extra:
        train_data = svhn.SVHN(root=svhn_path, split="train",
                                 transform=trn.Compose(
                                     [  # trn.Resize(32),
                                         trn.ToTensor(), trn.Normalize(mean, std)]))
    else:
        train_data = svhn.SVHN(root=svhn_path, split="train_and_extra",
                               transform=trn.Compose(
                                   [  # trn.Resize(32),
                                       trn.ToTensor(), trn.Normalize(mean, std)]))

    test_data = svhn.SVHN(root=svhn_path, split="test",
                              transform=trn.Compose(
                                  [  # trn.Resize(32),
                                      trn.ToTensor(), trn.Normalize(mean, std)]))

    train_data.targets = train_data.targets.astype('int64')
    test_data.targets = test_data.targets.astype('int64')
    return train_data, test_data


def load_dataset(dataset):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if dataset == 'lsun_c':
        print('loading LSUN_C')
        out_data = dset.ImageFolder(root=lsun_c_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std),
                                                           trn.RandomCrop(32, padding=4)]))

    if dataset == 'lsun_r':
        print('loading LSUN_R')
        out_data = dset.ImageFolder(root=lsun_r_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if dataset == 'isun':
        print('loading iSUN')
        out_data = dset.ImageFolder(root=isun_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    if dataset == 'dtd':
        print('loading DTD')
        out_data = dset.ImageFolder(root=dtd_path,
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
    if dataset == 'places':
        print('loading Places365')
        out_data = dset.ImageFolder(root=places_path,
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))

    return out_data


def load_in_data(in_dset, rng):

    train_data_in_orig, test_in_data = load_CIFAR(in_dset)


    idx = np.array(range(len(train_data_in_orig)))
    rng.shuffle(idx)
    train_len = int(0.5 * len(train_data_in_orig))
    train_idx = idx[:train_len]
    aux_idx = idx[train_len:]

    train_in_data = torch.utils.data.Subset(train_data_in_orig, train_idx)
    aux_in_data = torch.utils.data.Subset(train_data_in_orig, aux_idx)

    return train_in_data, aux_in_data, test_in_data


def load_out_data(aux_out_dset, test_out_dset, in_dset, rng, classes=[]):
    if aux_out_dset == test_out_dset:
        if aux_out_dset == 'tinyimages_300k':
            out_data = load_tinyimages_300k(in_dset)

            idx = np.array(range(len(out_data)))
            rng.shuffle(idx)
            train_len = int(0.99 * len(out_data))
            aux_subset_idxs = idx[:train_len]
            test_subset_idxs = idx[train_len:]

            aux_out_data = torch.utils.data.Subset(out_data, aux_subset_idxs)
            test_out_data = torch.utils.data.Subset(out_data, test_subset_idxs)

        else:
            out_data = load_dataset(aux_out_dset)

            idx = np.array(range(len(out_data)))
            rng.shuffle(idx)
            train_len = int(0.7 * len(out_data))
            aux_subset_idxs = idx[:train_len]
            test_subset_idxs = idx[train_len:]

            aux_out_data = torch.utils.data.Subset(out_data, aux_subset_idxs)
            test_out_data = torch.utils.data.Subset(out_data, test_subset_idxs)

    elif aux_out_dset != test_out_dset:
        # load aux data
        if aux_out_dset == 'tinyimages_300k':
            aux_out_data = load_tinyimages_300k()
        elif aux_out_dset == 'svhn':
            aux_out_data, _ = load_SVHN()
        elif aux_out_dset in ['cifar10', 'cifar100']:
            aux_out_data, _ = load_CIFAR(aux_out_dset)
        else:
            aux_out_data = load_dataset(aux_out_dset)

        # load test data
        if test_out_dset == 'svhn':
            _, test_out_data = load_SVHN()
        elif test_out_dset in ['cifar10', 'cifar100']:
            _, test_out_data = load_CIFAR(test_out_dset)
        else:
            test_out_data = load_dataset(test_out_dset)

    return aux_out_data, test_out_data


def train_valid_split(in_data, aux_in_data, aux_out_data, rng):
    '''

    Args:
        in_data: data from in-distribution, from test set
        aux_in_data: data from in-distribution component of mixture, not in test set
        aux_out_data: data from auxiliary dataset component of mixture, not in test set

    Returns:
        6 datasets: each dataset split into two, one for training (or testing) and the other for validation
    '''

    #create validation dataset for clean in distribution
    in_valid_size = min(int(0.3 * len(in_data)), 10000)

    idx = np.array(range(len(in_data)))
    rng.shuffle(idx)
    train_in_idx = idx[in_valid_size:]
    valid_in_idx = idx[:in_valid_size]

    test_in_data = torch.utils.data.Subset(in_data, train_in_idx)
    valid_in_data_final = torch.utils.data.Subset(in_data, valid_in_idx)

    #create validation dataset for in-distribution component of mixture
    aux_in_valid_size =  min(int(0.3 * len(aux_in_data)), 10000)

    idx = np.array(range(len(aux_in_data)))
    rng.shuffle(idx)
    train_aux_in_idx = idx[aux_in_valid_size:]
    valid_aux_in_idx = idx[:aux_in_valid_size]

    train_aux_in_data_final = torch.utils.data.Subset(aux_in_data, train_aux_in_idx)
    valid_aux_in_data_final = torch.utils.data.Subset(aux_in_data, valid_aux_in_idx)

    #create validation dataset for auxiliary dataset componenet of mixture
    aux_valid_size = min(int(0.3 * len(aux_out_data)), 10000)

    idx = np.array(range(len(aux_out_data)))
    rng.shuffle(idx)
    train_aux_out_idx = idx[aux_valid_size:]
    valid_aux_out_idx = idx[:aux_valid_size]

    train_aux_out_data_final = torch.utils.data.Subset(aux_out_data, train_aux_out_idx)
    valid_aux_out_data_final = torch.utils.data.Subset(aux_out_data, valid_aux_out_idx)

    return test_in_data, valid_in_data_final, train_aux_in_data_final, valid_aux_in_data_final, train_aux_out_data_final, valid_aux_out_data_final


def make_datasets(in_dset, aux_out_dset, test_out_dset, pi, state):
    # random seed
    rng = np.random.default_rng(state['seed'])

    print('building datasets...')

    train_in_data, aux_in_data, test_in_data = load_in_data(in_dset, pi, rng)
    aux_out_data, test_out_data = load_out_data(aux_out_dset, test_out_dset, in_dset, rng)

    # make validation set from CIFAR test set
    train_in_data_final = train_in_data
    test_in_data, valid_in_data_final, train_aux_in_data_final, valid_aux_in_data_final, train_aux_out_data_final, valid_aux_out_data_final = train_valid_split(
                                                                                                            test_in_data, aux_in_data, aux_out_data, rng
                                                                                                        )

    # create the dataloaders
    train_loader_in = torch.utils.data.DataLoader(
        train_in_data_final,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True)

    # validation for P_0
    valid_loader_in = torch.utils.data.DataLoader(
        valid_in_data_final,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    # auxiliary dataset

    #for in-distribution component of mixture
    #drop last batch to eliminate unequal batch size issues
    train_loader_aux_in = torch.utils.data.DataLoader(
        train_aux_in_data_final,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    valid_loader_aux_in = torch.utils.data.DataLoader(
        valid_aux_in_data_final,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    #for out-distribution component of mixture
    train_loader_aux_out = torch.utils.data.DataLoader(
        train_aux_out_data_final,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    valid_loader_aux_out = torch.utils.data.DataLoader(
        valid_aux_out_data_final,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    # test data for P_0
    test_loader = torch.utils.data.DataLoader(
        test_in_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    # test loader for ood
    test_loader_out = torch.utils.data.DataLoader(
        test_out_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    return train_loader_in, train_loader_aux_in, train_loader_aux_out, test_loader, test_loader_out, valid_loader_in, valid_loader_aux_in, valid_loader_aux_out


def make_test_dataset(in_data, test_out_dset, state):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    # load in-distribution data
    if in_data == 'cifar10':
        test_in_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform)
    elif in_data == 'cifar100':
        test_in_data = dset.CIFAR100(
            cifar100_path, train=False, transform=test_transform)

    #load out-distribution
    if test_out_dset == 'svhn':
        test_out_data = svhn.SVHN(root=svhn_path, split="test",
                                  transform=trn.Compose(
                                      [  # trn.Resize(32),
                                          trn.ToTensor(), trn.Normalize(mean, std)]), download=True)

    if test_out_dset == 'lsun_c':
        test_out_data = dset.ImageFolder(root=lsun_c_path,
                                         transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std), trn.RandomCrop(32, padding=4)]))

    if test_out_dset == 'lsun_r':
        test_out_data = dset.ImageFolder(root=lsun_r_path,
                                         transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if test_out_dset == 'isun':
        test_out_data = dset.ImageFolder(root=isun_path,
                                         transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if test_out_dset == 'dtd':
        test_out_data = dset.ImageFolder(root=dtd_path,
                                         transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))

    if test_out_dset == 'places':
        test_out_data = dset.ImageFolder(root=places_path,
                                         transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))

    #test data for P_0
    test_loader = torch.utils.data.DataLoader(
        test_in_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    # test loader for ood
    test_loader_ood = torch.utils.data.DataLoader(
        test_out_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    return test_loader, test_loader_ood
