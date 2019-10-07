#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import pandas as pd
import numpy as np
import random
import itertools
import math
import os

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import sklearn
from sklearn.decomposition import PCA

from neural_network import Attention_Net
from neural_network import Linear_Net
from datasets import FoodDataset

########################### FUNCTIONS
def match_output(output,label):
    #print(torch.max(output).item())
    #print(output[0].dot(label[0]).item())
    if torch.max(output).item() == output[0].dot(label[0]).item():
        return 1
    else:
        return 0


########################### PARAMS
#Constant
ADAM = "Adam"
SGD = "SGD"
L0 = "L0"
L1 = "L1"
L2 = "L2"
MSE = "MSE"
CEL = "CEL"
WD = "0005"
ATTENTION = "attention_net"
LINEAR = "linear_net"
RELU = "relu"
SIGMOID = "sigmoid"
DATE = "20191007"

## TRAIN PARAMS
K_FOLDER = 5
NET = ATTENTION
BATCH_SIZE = 5
LEARNING_RATE = 0.05
WEIGHT_DECAY = torch.tensor(0.005).float()
QUERY_DIM = 9
KEY_DIM = 6
FEATURE_DIM = 5
EPOCH = 1000
MOMENTUM = 0.9
REG = L2
ACT = SIGMOID
OPTIMIZER = SGD
BETAS = (0.9,0.999)
LOSS = CEL
MASK = True



if __name__ == '__main__':

    username = "Artificial"

    extra = "Data_1000_Epoch_" + str(DATE) + "_" + str(EPOCH) + "_Net_" + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + "_MASK_" + str(MASK)

    input_csv = "/home/li/food/data/20190922_limofei_1000_input.csv"
    output_csv = "/home/li/food/data/20190922_limofei_1000_output.csv"
    dataset = FoodDataset(input_csv,output_csv)

    plot_path = "/home/li/food/plot/" + str(DATE) + "/CV/"
    model_path = "/home/li/food/model/"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    params = (QUERY_DIM,KEY_DIM,FEATURE_DIM)
    net_list = []
    for i in range(K_FOLDER):
        model_file = model_path + "Data_1000_Epoch_20191004_1000_Net_attention_net_u_li_mofei_Q_9_K_6_F_5_REG_L2_ACT_sigmoid_WD_0005_MASK_True_CV_Model_" + str(i) + ".model"
        if NET == ATTENTION:
            net = Attention_Net(dataset,params,activation = ACT)
            net_list.append(net)

    dataloader = DataLoader(dataset = dataset,
                            batch_size = 1,
                            shuffle = True,
                            num_workers = 0)

    
    

    for k in range(K_FOLDER):
        net = net_list[k]
        valid_dist_list = []
        for im,label in dataloader:
            out,dist_origin = net.forward(im)
            for dist in dist_origin:
                valid_dist_list.append(list(dist.detach().numpy()))

        pca = PCA(n_components = 'mle')
        pca.fit(valid_dist_list)
        valid_feature = pca.transform(valid_dist_list)
        print("Model " + str(k))
        print(pca.explained_variance_ratio_)

        figure = "PCA_valid_Model_" + str(k)
        print(len(valid_feature[:,0]))
        plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
        plt.scatter(valid_feature[:,0], valid_feature[:,1])
        plt.grid()
        plt.xlim(-0.3,0.3)
        plt.ylim(-0.3,0.3)
        plt.savefig(plt_file)
        plt.close('all')
    
    
