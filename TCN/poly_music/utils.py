from scipy.io import loadmat
import torch
import numpy as np


def data_generator(dataset):
    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./mdata/JSB_Chorales.mat')
    elif dataset == "Muse":
        print('loading Muse data...')
        data = loadmat('./mdata/MuseData.mat')
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat('./mdata/Nottingham.mat')
    elif dataset == "Piano":
        print('loading Piano data...')
        data = loadmat('./mdata/Piano_midi.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test