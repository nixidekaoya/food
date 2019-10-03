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
DATE = "20190924"

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
    ############### Data Preparation ##############
    username = "li_mofei"

    extra = "Data_1000_Epoch_" + str(DATE) + "_" + str(EPOCH) + "_Net_" + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + "_MASK_" + str(MASK)
    #model_path = "/home/li/food/model/" + str(extra) + ".model"
    
    input_csv = "/home/li/food/data/20190922_limofei_1000_input.csv"
    output_csv = "/home/li/food/data/20190922_limofei_1000_output.csv"

    #valid_input_csv = "/home/li/food/data/20190903_limofei_100_input_validation.csv"
    #valid_output_csv = "/home/li/food/data/20190903_limofei_100_output_validation.csv"

    dataset = FoodDataset(input_csv,output_csv)

    #valid_dataset = FoodDataset(valid_input_csv, valid_output_csv)

    plot_path = "/home/li/food/plot/" + str(DATE) + "/CV/"
    train_log_path = plot_path + "train_log/"
    train_log_file_path = train_log_path + str(extra) + ".txt"
    


    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    if not os.path.exists(train_log_path):
        os.mkdir(train_log_path)
    

    data_num = dataset.data_num
    valid_data_num = int(data_num/K_FOLDER)
    train_data_num = int(data_num - valid_data_num)
    sample_data_num = int(valid_data_num)

    print(train_data_num)
    print(valid_data_num)

    splits_list = []
    for i in range(K_FOLDER):
        splits_list.append(sample_data_num)
    splits_list = tuple(splits_list)
    print(splits_list)

    datasets = torch.utils.data.random_split(dataset,splits_list)
    dataloader_list = []
    dataloader_bs1_list = []
    #### DEBUG
    #for ds in datasets:
        #print(ds.__len__())

    for ds in datasets:
        dataloader = DataLoader(dataset = ds,
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers = 0)
        dataloader_list.append(dataloader)

        dataloader_bs1 = DataLoader(dataset = ds,
                                    batch_size = 1,
                                    shuffle = True,
                                    num_workers = 0)
        dataloader_bs1_list.append(dataloader_bs1)

        

    params = (QUERY_DIM,KEY_DIM,FEATURE_DIM)

    net_list = []
    ## NET

    if NET == ATTENTION:
        for i in range(K_FOLDER):
            net = Attention_Net(dataset, params, activation = ACT)
            net_list.append(net)
    elif NET == LINEAR:
        for i in range(K_FOLDER):
            net = Linear_Net(dataset, params = FEATURE_DIM, activation = ACT)
            net_list.append(net)



    optimizer_list = []
    ## OPTIMIZER
    if OPTIMIZER == SGD:
        for i in range(K_FOLDER):
            optimizer = torch.optim.SGD(net_list[i].parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
            optimizer_list.append(optimizer)
    elif OPTIMIZER == ADAM:
        for i in range(K_FOLDER):
            optimizer = torch.optim.Adam(net_list[i].parameters(), lr = LEARNING_RATE, betas = BETAS)
            optimizer_list.append(optimizer)
            
    ## LOSS
    if LOSS == MSE:
        loss_function = torch.nn.MSELoss()
    elif LOSS == CEL:
        loss_function = torch.nn.CrossEntropyLoss()

    ########## TRAINING
    train_loss_list = []
    train_loss_log_list = []
    valid_loss_list = []
    valid_loss_log_list = []
    valid_accurate_rate_list = []
    train_accurate_rate_list = []

    train_average_accuracy_list = []
    valid_average_accuracy_list = []

    for k in range(K_FOLDER):
        train_loss_list.append([])
        train_loss_log_list.append([])
        valid_loss_list.append([])
        valid_loss_log_list.append([])
        valid_accurate_rate_list.append([])
        train_accurate_rate_list.append([])
    

    for epoch in range(EPOCH):
        #dist_list = []
        #valid_dist_list = []
        #test_output_list = []
        train_average_accuracy = 0
        valid_average_accuracy = 0
        for k in range(K_FOLDER):
            valid_dataloader = dataloader_list[k]
            train_dataloader = dataloader_list[:k] + dataloader_list[k+1:]

            valid_dataloader_bs1 = dataloader_bs1_list[k]
            train_dataloader_bs1 = dataloader_bs1_list[:k] + dataloader_bs1_list[k+1:]
            net = net_list[k]
            optimizer = optimizer_list[k]
        
            net.train()

            train_loss_list_each_epoch = []
            valid_loss_list_each_epoch = []

            for dataloader in train_dataloader:
                for im,label in dataloader:
                    l0_regularization = torch.tensor(0).float()
                    l1_regularization = torch.tensor(0).float()
                    l2_regularization = torch.tensor(0).float()

                    if NET == ATTENTION:
                        out,dist_origin = net.forward(im,masked = MASK)
                        #for dist in dist_origin:
                            #dist_list.append(list(dist.detach().numpy()))
                    elif NET == LINEAR:
                        out = net.forward(im)

                    org_loss = loss_function(out, torch.max(label,1)[1])

                    ## Regularization
                    for param in net.parameters():
                        l1_regularization += WEIGHT_DECAY * torch.norm(param,1)
                        l2_regularization += WEIGHT_DECAY * torch.norm(param,2)

                    if REG == L0:
                        loss = org_loss + l0_regularization
                    elif REG == L1:
                        loss = org_loss + l1_regularization
                    elif REG == L2:
                        loss = org_loss + l2_regularization

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    

            accurate_number = 0
            #counter = 0
            for dataloader in train_dataloader_bs1:
                train_loss_each = 0
                    
                for im,label in dataloader:
                    #counter += 1
                    if NET == ATTENTION:
                        out,dist_origin = net.forward(im)
                        accurate_number += match_output(out,label.float())
                        #print(accurate_number)
                    elif NET == LINEAR:
                        out = net.forward(im,mode="valid")
                        accurate_number += match_output(out,label.float())
                        
                    org_loss = loss_function(out, torch.max(label,1)[1])
                    train_loss_list_each_epoch.append(org_loss.item())
            #print(counter)

            train_accurate_rate = float(accurate_number)/ train_data_num
            train_accurate_rate_list[k].append(train_accurate_rate)
            train_average_accuracy += train_accurate_rate
            train_loss = np.mean(train_loss_list_each_epoch)
            train_loss_list[k].append(train_loss)
            train_loss_log_list[k].append(math.log(train_loss))

            accurate_number = 0
            for im,label in valid_dataloader_bs1:
                if NET == ATTENTION:
                    out,dist_origin = net.forward(im)
                    accurate_number += match_output(out,label.float())
                elif NET == LINEAR:
                    out = net.forward(im,mode="valid")
                    accurate_number += match_output(out,label.float())

                    org_loss = loss_function(out, torch.max(label,1)[1])
                    valid_loss_list_each_epoch.append(org_loss.item())

            valid_accurate_rate = float(accurate_number)/ valid_data_num
            valid_accurate_rate_list[k].append(valid_accurate_rate)
            valid_average_accuracy += valid_accurate_rate
            valid_loss = np.mean(valid_loss_list_each_epoch)
            valid_loss_list[k].append(valid_loss)
            valid_loss_log_list[k].append(math.log(valid_loss))


            ### Train LOG
            info1 = "Epoch: " + str(epoch) + " , CV_Model: " + str(k) + " , Train Loss: " + str(train_loss)
            info2 = "Epoch: " + str(epoch) + " , CV_Model: " + str(k) + " , valid Loss: " + str(valid_loss)
            info3 = "Epoch: " + str(epoch) + " , CV_Model: " + str(k) + " , Train Accuracy: " + str(train_accurate_rate) + " , Test Accuracy: " + str(valid_accurate_rate) 
            print(info3)
            info5 = "Epoch: " + str(epoch) + " , Output: " + str(out)
            #print(info5)
            for para in net.parameters():
                info6 = "Epoch: " + str(epoch) + " , Parameters: " + str(para)
                #print(info6)
            

        train_average_accuracy_list.append(train_average_accuracy/K_FOLDER)
        valid_average_accuracy_list.append(valid_average_accuracy/K_FOLDER)

        

    #### Model Save
    #print(model_path)
    for k in range(K_FOLDER):
        model_path = "/home/li/food/model/" + str(extra) + "_CV_Model_" + str(k) + ".model"
        torch.save(net_list[k].state_dict(), model_path)


    #### PLOT
    figure = "Learning_Curve"
    for k in range(K_FOLDER):
        plt_file = plot_path + str(extra) + "_" + str(figure) + "_CV_Model_" + str(k) + ".png"
        #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
        plt.plot(range(len(train_loss_log_list[k])), train_loss_log_list[k], label = "log train loss")
        plt.plot(range(len(valid_loss_log_list[k])), valid_loss_log_list[k], label = "log valid loss")    
        plt.legend(loc = "upper right")
        plt.savefig(plt_file)
        plt.close('all')


    figure = "Accurate_Rate_Curve"
    for k in range(K_FOLDER):
        plt_file = plot_path + str(extra) + "_" + str(figure) + "_CV_Model_" + str(k) + ".png"
        plt.plot(range(len(valid_accurate_rate_list[k])), valid_accurate_rate_list[k], label = "Valid Accurate Rate")
        plt.plot(range(len(train_accurate_rate_list[k])), train_accurate_rate_list[k], label = "Train Accurate Rate")
        plt.legend(loc = "lower right")
        plt.savefig(plt_file)
        plt.close('all')

    figure = "Accuracy_Average_Curve" 
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    plt.plot(range(len(valid_average_accuracy_list)), valid_average_accuracy_list, label = "Valid Accurate Average")
    plt.plot(range(len(train_average_accuracy_list)), train_average_accuracy_list, label = "Train Accurate Average")
    plt.legend(loc = "lower right")
    plt.savefig(plt_file)
    plt.close('all')

    figure = "Accuracy_Scatter"
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    x = ["Train Accuracy","Test Accuracy"]
    for k in range(K_FOLDER):
        y = []
        y.append(train_accurate_rate_list[k][-1])
        y.append(valid_accurate_rate_list[k][-1])
        label = "Model_" + str(k)
        plt.scatter(x,y,s = 600, c = "blue", alpha = float(1)/K_FOLDER * (k+1), linewidths = "2", marker = "o", label = label)

    y = []
    y.append(train_average_accuracy_list[-1])
    y.append(valid_average_accuracy_list[-1])
    plt.scatter(x,y,s = 600, c = "yellow", alpha = 1, linewidths = "2", marker = "*", label = "Average")
    plt.legend(loc = "upper right")
    plt.savefig(plt_file)
    plt.close('all')
    
    
        
        
    
        
