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
WD = "000001"
ATTENTION = "attention_net"
LINEAR = "linear_net"
RELU = "relu"
SIGMOID = "sigmoid"
DATE = "201909010"

## TRAIN PARAMS
NET = LINEAR
BATCH_SIZE = 10
LEARNING_RATE = 0.05
WEIGHT_DECAY = torch.tensor(0.000001).float()
QUERY_DIM = 9
KEY_DIM = 6
FEATURE_DIM = 5
EPOCH = 5000
MOMENTUM = 0.9
REG = L0
ACT = SIGMOID
OPTIMIZER = SGD
BETAS = (0.9,0.999)
LOSS = MSE


if __name__ == '__main__':
    ############### Data Preparation ##############
    username = "li_mofei"

    extra = "Data_500_Epoch_" + str(DATE) + "_" + str(EPOCH) + "_Net_" + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD)
    model_path = "/home/li/food/model/" + str(extra) + ".model"
    train_log_path = "/home/li/food/model/train_log/" + str(extra) + ".txt"

    input_csv = "/home/li/food/data/20190910_limofei_500_input.csv"
    output_csv = "/home/li/food/data/20190910_limofei_500_output.csv"

    valid_input_csv = "/home/li/food/data/20190903_limofei_100_input_validation.csv"
    valid_output_csv = "/home/li/food/data/20190903_limofei_100_output_validation.csv"

    dataset = FoodDataset(input_csv,output_csv)

    valid_dataset = FoodDataset(valid_input_csv, valid_output_csv)

    plot_path = "/home/li/food/plot/" + str(DATE) + "/"

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    data_num = dataset.data_num
    valid_data_num = valid_dataset.data_num

    dataloader = DataLoader(dataset = dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = True,
                            num_workers = 0)


    valid_dataloader = DataLoader(dataset = valid_dataset,
                                  batch_size = 1,
                                  shuffle = True,
                                  num_workers = 0)

    params = (QUERY_DIM,KEY_DIM,FEATURE_DIM)

    ## NET

    if NET == ATTENTION:
        net = Attention_Net(dataset, params, activation = ACT)
    elif NET == LINEAR:
        net = Linear_Net(dataset, params = FEATURE_DIM, activation = ACT)

    ## OPTIMIZER
    if OPTIMIZER == SGD:
        optimizer = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    elif OPTIMIZER == ADAM:
        optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE, betas = BETAS)

    ## LOSS
    loss_function = torch.nn.MSELoss()

    ########## TRAINING
    train_loss_list = []
    train_loss_log_list = []
    test_loss_list = []
    test_loss_log_list = []
    test_accurate_rate_list = []
    

    for epoch in range(EPOCH):
        net.train()
        dist_list = []
        valid_dist_list = []
        test_output_list = []
        accurate_number = 0
        

        train_loss_list_each_epoch = []
        test_loss_list_each_epoch = []

        for im,label in dataloader:
            l0_regularization = torch.tensor(0).float()
            l1_regularization = torch.tensor(0).float()
            l2_regularization = torch.tensor(0).float()

            if NET == ATTENTION:
                out,dist_origin = net.forward(im,mode="train")
                for dist in dist_origin:
                    dist_list.append(list(dist.detach().numpy()))
            elif NET == LINEAR:
                out = net.forward(im,mode="train")

            mse_loss = loss_function(out, label)

            ## Regularization
            for param in net.parameters():
                l1_regularization += WEIGHT_DECAY * torch.norm(param,1)
                l2_regularization += WEIGHT_DECAY * torch.norm(param,2)

            if REG == L0:
                loss = mse_loss + l0_regularization
            elif REG == L1:
                loss = mse_loss + l1_regularization
            elif REG == L2:
                loss = mse_loss + l2_regularization

            train_loss_list_each_epoch.append(mse_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for im,label in valid_dataloader:
            if NET == ATTENTION:
                out,dist_origin = net.forward(im,mode="valid")
                for dist in dist_origin:
                    valid_dist_list.append(list(dist.detach().numpy()))
                output_array = list(out.detach().numpy())
                test_output_list.append(output_array)
                accurate_number += match_output(out,label)
            elif NET == LINEAR:
                out = net.forward(im,mode="valid")
                output_array = list(out.detach().numpy())
                test_output_list.append(output_array)
                accurate_number += match_output(out,label)

            mse_loss = loss_function(out, label)
            test_loss_list_each_epoch.append(mse_loss.item())

        accurate_rate = float(accurate_number)/ valid_data_num
        test_accurate_rate_list.append(accurate_rate)

        train_loss = np.mean(train_loss_list_each_epoch)
        train_loss_list.append(train_loss)
        train_loss_log_list.append(math.log(train_loss))
        test_loss = np.mean(test_loss_list_each_epoch)
        test_loss_list.append(test_loss)
        test_loss_log_list.append(math.log(test_loss))


        info1 = "Epoch: " + str(epoch) + " , Train Loss: " + str(train_loss)
        info2 = "Epoch: " + str(epoch) + " , Test Loss: " + str(test_loss)
        print(info1)
        print(info2)
        if NET == ATTENTION:
            info3 = "Epoch: " + str(epoch) + " , Distribution: " + str(dist_origin)
            print(info3)
        info4 = "Epoch: " + str(epoch) + " , Accurate Rate: " + str(accurate_rate)
        print(info4)


    print(model_path)
    torch.save(net.state_dict(), model_path)

    model = net

    #### PLOT
    figure = "Learning_Curve" 
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    plt.plot(range(len(train_loss_log_list)), train_loss_log_list, label = "log train loss")
    plt.plot(range(len(test_loss_log_list)), test_loss_log_list, label = "log test loss")    
    plt.legend(loc = "upper right")
    plt.savefig(plt_file)
    plt.close('all')


    figure = "Accurate_Rate_Curve" 
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    plt.plot(range(len(test_accurate_rate_list)), test_accurate_rate_list, label = "Accurate Rate")
    plt.legend(loc = "upper right")
    plt.savefig(plt_file)
    plt.close('all')

    ##### TEST

    if NET == ATTENTION:
        pca = PCA(n_components = 'mle')
        pca.fit(dist_list)
    
        feature = pca.transform(dist_list)
        print(pca.explained_variance_ratio_)
    
        figure = "PCA_train"
        plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
        plt.scatter(feature[:,0], feature[:,1])
        plt.grid()
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.savefig(plt_file)
        plt.close('all')

        pca_valid = PCA(n_components = 'mle')
        pca_valid.fit(valid_dist_list)
        valid_feature = pca_valid.transform(valid_dist_list)
        print(pca_valid.explained_variance_ratio_)

        figure = "PCA_valid"
        plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
        plt.scatter(valid_feature[:,0], valid_feature[:,1])
        plt.grid()
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.savefig(plt_file)
        plt.close('all')


    

    with open(train_log_path,"w") as log_f:
        log_f.write(info1 + "\r\n")
        log_f.write(info2 + "\r\n")
        if NET == ATTENTION:
            log_f.write(info3 + "\r\n")
        log_f.write(info4 + "\r\n")
        if NET == ATTENTION:
            log_f.write("Variance Ratio Train:" + str(pca.explained_variance_ratio_) + "\r\n")
            log_f.write("Variance Ratio Test:" + str(pca_valid.explained_variance_ratio_) + "\r\n")
        
        
    
        
