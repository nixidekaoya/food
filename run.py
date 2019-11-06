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
from neural_network import Linear_NoHidden_Net
from datasets import FoodDataset

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
LINEAR_NOHIDDEN = "linear_no_hidden_net"
RELU = "relu"
SIGMOID = "sigmoid"
DATE = "20191010"


########################### FUNCTIONS
def match_output(output,label):
    #print(torch.max(output).item())
    #print(output[0].dot(label[0]).item())
    if torch.max(output).item() == output[0].dot(label[0]).item():
        return 1
    else:
        return 0
    
def calculate_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a,b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot/(norma * normb)
    return cos



def run_normal(input_csv,output_csv,weight_csv,username,ARTIFICIAL,BATCH_NORM,DATE,EPOCH,KEY_DIM,FEATURE_DIM,QUERY_DIM,REG,LEARNING_RATE,WEIGHT_DECAY,LOSS,ACT,BATCH_SIZE,OPTIMIZER,NET,w_f,w_f_type,WD,MOMENTUM,extra_msg):

    ############### Data Preparation ##############
    extra = str(extra_msg) + str(DATE) + "_Epoch_"+ str(EPOCH) + "_Net_" + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "WD" + str(WD) + "_wf_" + str(w_f_type)
    
    model_path = "/home/li/food/model/" + str(extra) + ".model"

    valid_input_csv = "/home/li/food/data/20190903_limofei_100_input_validation.csv"
    valid_output_csv = "/home/li/food/data/20190903_limofei_100_output_validation.csv"
    valid_dataset = FoodDataset(valid_input_csv, valid_output_csv)
    valid_dataloader = DataLoader(dataset = valid_dataset,
                                batch_size = 1,
                                shuffle = False,
                                num_workers = 0)

    valid_data_num = valid_dataset.data_num
    dataset = FoodDataset(input_csv,output_csv)
    plot_path = "/home/li/food/plot/" + str(DATE) + "/"
    train_log_path = plot_path + "train_log/"
    train_log_file_path = train_log_path + str(extra) + ".txt"
    
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    if not os.path.exists(train_log_path):
        os.mkdir(train_log_path)
    
    data_num = dataset.data_num
    train_data_num = data_num

    dataloader = DataLoader(dataset = dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = 0)

    train_dataloader = DataLoader(dataset = dataset,
                                  batch_size = 1,
                                  shuffle = False,
                                  num_workers = 0)

    params = (QUERY_DIM,KEY_DIM,FEATURE_DIM)

    ## NET

    if NET == ATTENTION:
        if w_f == "FIXED":
            net = Attention_Net(dataset, params, activation = ACT, w_f = "Fixed", w_f_type = w_f_type)
        elif w_f == "TRAIN":
            net = Attention_Net(dataset, params, activation = ACT, w_f = "Train")
        
    elif NET == LINEAR:
        net = Linear_Net(dataset, params = FEATURE_DIM, activation = ACT)


    #for i in filter(lambda p: p.requires_grad, net.parameters()):
    #    print(i)
    #for i in filter(lambda p: not p.requires_grad, net.parameters()):
    #    print(i)


    ## OPTIMIZER
    if OPTIMIZER == SGD:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = LEARNING_RATE, momentum = MOMENTUM)
        #optimizer = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    elif OPTIMIZER == ADAM:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LEARNING_RATE, betas = BETAS)

    ## LOSS
    if LOSS == MSE:
        loss_function = torch.nn.MSELoss()
    elif LOSS == CEL:
        loss_function = torch.nn.CrossEntropyLoss()


    ########## TRAINING
    train_loss_list = []
    train_loss_log_list = []
    train_accurate_rate_list = []

    if not ARTIFICIAL:
        test_loss_list = []
        test_loss_log_list = []
        test_accurate_rate_list = []
    

    for epoch in range(EPOCH):
        net.train()
        dist_list = []
        valid_dist_list = []

        if not ARTIFICIAL:
            test_output_list = []
            test_loss_list_each_epoch = []
            
        train_loss_list_each_epoch = []
        
        if not ARTIFICIAL:
            accurate_number = 0
            for im,label in valid_dataloader:
                if NET == ATTENTION:
                    out,dist_origin = net.forward(im,batch_norm = BATCH_NORM)
                    for dist in dist_origin:
                        valid_dist_list.append(list(dist.detach().numpy()))
                    output_array = list(out.detach().numpy())
                    test_output_list.append(output_array)
                    accurate_number += match_output(out,label.float())
                elif NET == LINEAR:
                    out = net.forward(im,batch_norm = BATCH_NORM)
                    output_array = list(out.detach().numpy())
                    test_output_list.append(output_array)
                    accurate_number += match_output(out,label.float())

                org_loss = loss_function(out, torch.max(label,1)[1])
                test_loss_list_each_epoch.append(org_loss.item())

            valid_accurate_rate = float(accurate_number) / valid_data_num
            test_accurate_rate_list.append(valid_accurate_rate)

            
        if not ARTIFICIAL:
            test_loss = np.mean(test_loss_list_each_epoch)
            test_loss_list.append(test_loss)
            test_loss_log_list.append(math.log(test_loss))

        accurate_number = 0
        for im,label in train_dataloader:
            if NET == ATTENTION:
                out,dist_origin = net.forward(im,batch_norm = BATCH_NORM)
                for dist in dist_origin:
                    dist_list.append(list(dist.detach().numpy()))
                output_array = list(out.detach().numpy())
                #test_output_list.append(output_array)
                accurate_number += match_output(out,label.float())
            elif NET == LINEAR:
                out = net.forward(im,batch_norm = BATCH_NORM)
                output_array = list(out.detach().numpy())
                #test_output_list.append(output_array)
                accurate_number += match_output(out,label.float())

            #org_loss = loss_function(out, torch.max(label,1)[1])
            #test_loss_list_each_epoch.append(org_loss.item())

        train_accurate_rate = float(accurate_number)/ train_data_num
        train_accurate_rate_list.append(train_accurate_rate)

        train_loss = np.mean(train_loss_list_each_epoch)
        train_loss_list.append(train_loss)
        train_loss_log_list.append(math.log(train_loss))
            
        for im,label in dataloader:
            l0_regularization = torch.tensor(0).float()
            l1_regularization = torch.tensor(0).float()
            l2_regularization = torch.tensor(0).float()

            if NET == ATTENTION:
                out,dist_origin = net.forward(im,masked = MASK,batch_norm = BATCH_NORM)
            elif NET == LINEAR:
                out = net.forward(im,batch_norm = BATCH_NORM)
            elif NET == LINEAR_NOHIDDEN:
                out = net.forward(im,batch_norm = BATCH_NORM)

            #print(out)
            #print(label)
            #print(out.shape)
            #print(label.shape)

            org_loss = loss_function(out, torch.max(label,1)[1])
            #print(torch.max(label,1)[1])
            #print(org_loss)

            ## Regularization
            for param in filter(lambda p: p.requires_grad, net.parameters()):
                l1_regularization += WEIGHT_DECAY * torch.norm(param,1)
                l2_regularization += WEIGHT_DECAY * torch.norm(param,2)

            if REG == L0:
                loss = org_loss + l0_regularization
            elif REG == L1:
                loss = org_loss + l1_regularization
            elif REG == L2:
                loss = org_loss + l2_regularization

            train_loss_list_each_epoch.append(org_loss.item())

            optimizer.zero_grad()
            #print(loss)
            loss.backward()
            optimizer.step()

        info1 = "Epoch: " + str(epoch) + " , Train Loss: " + str(train_loss)
        print(info1)
        if not ARTIFICIAL:
            info2 = "Epoch: " + str(epoch) + " , Test Loss: " + str(test_loss)
            print(info2)

        if NET == ATTENTION:
            info3 = "Epoch: " + str(epoch) + " , Distribution: " + str(dist_origin)
            print(info3)
        info4 = "Epoch: " + str(epoch) + " , Train Accuracy: " + str(train_accurate_rate)
        print(info4)

        if not ARTIFICIAL:
            info5 = "Epoch: " + str(epoch) + " , Test Accuracy: " + str(valid_accurate_rate)
            print(info5)
        
        for para in net.parameters():
            info6 = "Epoch: " + str(epoch) + " , Parameters: " + str(para)
            #print(info6)

    print(model_path)
    torch.save(net.state_dict(), model_path)

    model = net

    #### PLOT
    figure = "Learning_Curve" 
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    plt.plot(range(len(train_loss_log_list)), train_loss_log_list, label = "log train loss")
    if not ARTIFICIAL:
        plt.plot(range(len(test_loss_log_list)), test_loss_log_list, label = "log test loss")    
    plt.legend(loc = "upper right")
    plt.savefig(plt_file)
    plt.close('all')


    figure = "Accurate_Rate_Curve" 
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    if not ARTIFICIAL:
        plt.plot(range(len(test_accurate_rate_list)), test_accurate_rate_list, label = "Valid Accurate Rate")
    plt.plot(range(len(train_accurate_rate_list)), train_accurate_rate_list, label = "Train Accurate Rate")
    
    plt.legend(loc = "lower right")
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
        print(len(valid_feature[:,0]))
        plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
        plt.scatter(valid_feature[:,0], valid_feature[:,1])
        plt.grid()
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.savefig(plt_file)
        plt.close('all')


    with open(train_log_file_path,"w") as log_f:
        log_f.write(info1 + "\r\n")
        log_f.write(info2 + "\r\n")
        if NET == ATTENTION:
            log_f.write(info3 + "\r\n")
        log_f.write(info4 + "\r\n")
        if NET == ATTENTION:
            log_f.write("Variance Ratio Train:" + str(pca.explained_variance_ratio_) + "\r\n")
            log_f.write("Variance Ratio Test:" + str(pca_valid.explained_variance_ratio_) + "\r\n")
    
    return

def run_cross_validation(input_csv,output_csv,weight_csv,username,K_FOLDER,BATCH_NORM,MOMENTUM,ARTIFICIAL,DATE,EPOCH,KEY_DIM,FEATURE_DIM,QUERY_DIM,REG,LEARNING_RATE,WEIGHT_DECAY,LOSS,ACT,BATCH_SIZE,OPTIMIZER,NET,w_f,w_f_type,VALIDATE_NUMBER,WD,extra_msg,ATTENTION_REG, ATTENTION_REG_WEIGHT):
        ############### Data Preparation ##############
    extra = "CV_" + str(extra_msg) + str(DATE) + "_Epoch_"+ str(EPOCH) + "_Net_" + str(NET) + "_u_" + str(username) + "_Q_" + str(QUERY_DIM) + "_K_" + str(KEY_DIM) + "_F_" + str(FEATURE_DIM) + "_REG_" + str(REG) + "_ACT_" + str(ACT) + "_WD_" + str(WD) + "w_f" + str(w_f_type)
    #model_path = "/home/li/food/model/" + str(extra) + ".model"

    ## Artificial
    dataset = FoodDataset(input_csv,output_csv)
    
    print(weight_csv)
    weights_df = pd.read_csv(weight_csv)


    model_path = "/home/li/food/model/" + str(DATE) + "/"
    
    plot_path = "/home/li/food/plot/" + str(DATE) + "/"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_path = plot_path + "CV/"
    train_log_path = plot_path + "train_log/"
    train_log_file_path = train_log_path + str(extra) + ".txt"
    
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    if not os.path.exists(model_path):
        os.mkdir(model_path)

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
            if w_f == "FIXED":
                net = Attention_Net(dataset, params, activation = ACT, w_f = "Fixed", w_f_type = w_f_type)
            elif w_f == "TRAIN":
                net = Attention_Net(dataset, params, activation = ACT, w_f = "Train")
            net_list.append(net)
    elif NET == LINEAR:
        for i in range(K_FOLDER):
            net = Linear_Net(dataset, params = KEY_DIM, activation = ACT)
            net_list.append(net)
    elif NET == LINEAR_NOHIDDEN:
        for i in range(K_FOLDER):
            net = Linear_NoHidden_Net(dataset)
            net_list.append(net)
            
    ### TEST
    
    for k in range(K_FOLDER):
        model_save = model_path + str(extra) + "_CV_Model_" + str(k) + "initial.model"
        torch.save(net_list[k].state_dict(), model_save)
        print(model_save)
    

    optimizer_list = []
    ## OPTIMIZER
    if OPTIMIZER == SGD:
        for i in range(K_FOLDER):
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net_list[i].parameters()), lr = LEARNING_RATE, momentum = MOMENTUM)
            optimizer_list.append(optimizer)
    elif OPTIMIZER == ADAM:
        for i in range(K_FOLDER):
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net_list[i].parameters()), lr = LEARNING_RATE, betas = BETAS)
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
        
            
            net.eval()
            
            
            train_loss_list_each_epoch = []
            valid_loss_list_each_epoch = []
            
            
            accurate_number = 0
            for im,label in valid_dataloader_bs1:
                if NET == ATTENTION:
                    out,dist_origin = net.forward(im,batch_norm = BATCH_NORM)
                    accurate_number += match_output(out,label.float())
                else:
                    out = net.forward(im,batch_norm = BATCH_NORM)
                    accurate_number += match_output(out,label.float())

                org_loss = loss_function(out, torch.max(label,1)[1])
                valid_loss_list_each_epoch.append(org_loss.item())
                    
            
            valid_accurate_rate = float(accurate_number)/ valid_data_num
            valid_accurate_rate_list[k].append(valid_accurate_rate)
            valid_average_accuracy += valid_accurate_rate
            valid_loss = np.mean(valid_loss_list_each_epoch)
            valid_loss_list[k].append(valid_loss)
            valid_loss_log_list[k].append(math.log(valid_loss))
            
                    
            accurate_number = 0
            #counter = 0
            for dataloader in train_dataloader_bs1:
                train_loss_each = 0
                    
                for im,label in dataloader:
                    #counter += 1
                    if NET == ATTENTION:
                        out,dist_origin = net.forward(im,batch_norm = BATCH_NORM)
                        accurate_number += match_output(out,label.float())
                        #print(accurate_number)
                    else:
                        out = net.forward(im,batch_norm = BATCH_NORM)
                        accurate_number += match_output(out,label.float())
                        
                    org_loss = loss_function(out, torch.max(label,1)[1])
                    train_loss_list_each_epoch.append(org_loss.item())
                                   
                                   
                                   
                                   
            train_accurate_rate = float(accurate_number)/ train_data_num
            train_accurate_rate_list[k].append(train_accurate_rate)
            train_average_accuracy += train_accurate_rate
            train_loss = np.mean(train_loss_list_each_epoch)
            train_loss_list[k].append(train_loss)
            train_loss_log_list[k].append(math.log(train_loss))

            net.train()
            
            for dataloader in train_dataloader:
                for im,label in dataloader:
                    l0_regularization = torch.tensor(0).float()
                    l1_regularization = torch.tensor(0).float()
                    l2_regularization = torch.tensor(0).float()
                    attention_regularization = torch.tensor(0).float()

                    if NET == ATTENTION:
                        out,dist_origin = net.forward(im,batch_norm = BATCH_NORM)
                        #for dist in dist_origin:
                            #dist_list.append(list(dist.detach().numpy()))
                        if ATTENTION_REG:
                            for dist in dist_origin:
                                attention_regularization += ATTENTION_REG_WEIGHT * torch.norm(dist,0.5)
                                #print(attention_regularization)
                                
                            
                    else:
                        out = net.forward(im,batch_norm = BATCH_NORM)

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
                    
                    if ATTENTION_REG:
                        loss = loss + attention_regularization

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            ### Train LOG
            info1 = "Epoch: " + str(epoch) + " , CV_Model: " + str(k) + " , Train Loss: " + str(train_loss)
            info2 = "Epoch: " + str(epoch) + " , CV_Model: " + str(k) + " , valid Loss: " + str(valid_loss)
            info3 = "Epoch: " + str(epoch) + " , CV_Model: " + str(k) + " , Train Accuracy: " + str(train_accurate_rate) + " , Test Accuracy: " + str(valid_accurate_rate) 
            #print(info3)
            info5 = "Epoch: " + str(epoch) + " , Output: " + str(out)
            #print(info5)
            if epoch % 50 == 0 or epoch == (EPOCH - 1):
                print(info3)
            for para in net.parameters():
                info6 = "Epoch: " + str(epoch) + " , Parameters: " + str(para)
                #print(info6)
            

        train_average_accuracy_list.append(train_average_accuracy/K_FOLDER)
        valid_average_accuracy_list.append(valid_average_accuracy/K_FOLDER)
        
        if epoch % 100 == 0:
            for k in range(K_FOLDER):
                model_save = model_path + str(extra) + "_CV_Model_" + str(k) + ".model"
                torch.save(net_list[k].state_dict(), model_save)

        

    #### Model Save
    #print(model_path)
    for k in range(K_FOLDER):
        model_save = model_path + str(extra) + "_CV_Model_" + str(k) + ".model"
        torch.save(net_list[k].state_dict(), model_save)
        print(model_save)


    #### PLOT
    
                                   
                                   
    figure = "Learning_Curve"
    for k in range(K_FOLDER):
        plt.figure()
        title = "F_" + str(FEATURE_DIM) + "_K_"+ str(KEY_DIM) + "_wf_" + str(w_f) + "_Model_" + str(k)
        plt_file = plot_path + str(extra) + "_" + str(figure) + "_CV_Model_" + str(k) + ".png"
        #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
        plt.plot(range(len(train_loss_log_list[k])), train_loss_log_list[k], label = "log train loss")
        plt.plot(range(len(valid_loss_log_list[k])), valid_loss_log_list[k], label = "log valid loss")
                                   
        plt.title(title)
        plt.legend(loc = "upper right")
        plt.show()
        plt.savefig(plt_file)
        plt.close('all')


    figure = "Accuracy_Curve"
    for k in range(K_FOLDER):
        plt.figure()
        title = "F_" + str(FEATURE_DIM) + "_K_"+ str(KEY_DIM) + "_wf_" + str(w_f) + "_Model_" + str(k)
        plt_file = plot_path + str(extra) + "_" + str(figure) + "_CV_Model_" + str(k) + ".png"
        plt.plot(range(len(valid_accurate_rate_list[k])), valid_accurate_rate_list[k], label = "Valid Accurate Rate")
        plt.plot(range(len(train_accurate_rate_list[k])), train_accurate_rate_list[k], label = "Train Accurate Rate")
        plt.title(title)
        plt.legend(loc = "lower right")
        plt.show()
        plt.savefig(plt_file)
        plt.close('all')

    figure = "Accuracy_Average_Curve"
    title = "F_" + str(FEATURE_DIM) + "_K_"+ str(KEY_DIM) + "_wf_" + str(w_f)
    plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
    plt.figure()
    #plt.plot(range(len(train_loss_list)), train_loss_list, label = "train loss")
    plt.plot(range(len(valid_average_accuracy_list)), valid_average_accuracy_list, label = "Valid Accurate Average")
    plt.plot(range(len(train_average_accuracy_list)), train_average_accuracy_list, label = "Train Accurate Average")
                                   
    plt.title(title)
    plt.legend(loc = "lower right")
    plt.savefig(plt_file)
    plt.show()
    plt.close('all')

    figure = "Accuracy_Scatter"
    plt.figure()
    
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
    plt.title(title)
    plt.legend(loc = "best")
    plt.show()
    plt.savefig(plt_file)
    plt.close('all')
    
    
    if NET != ATTENTION:
        return


    data_list = []
    counter = 0
    for data in dataloader_bs1_list[0]:
        counter += 1
        data_list.append(data)
        if counter >= VALIDATE_NUMBER:
            break

            
    
    
    for k in range(K_FOLDER):
        net = net_list[k]
        net.eval()
        valid_dist_list = []
        #valid_dataloader = dataloader_bs1_list[k]
        for data in data_list:
            out,dist_origin = net.forward(data[0],batch_norm = BATCH_NORM)
            for dist in dist_origin:
                valid_dist_list.append(list(dist.detach().numpy()))

        pca = PCA(n_components = 2)
        pca.fit(valid_dist_list)
        valid_feature = pca.transform(valid_dist_list)
        print("Model " + str(k))
        print(pca.explained_variance_ratio_)
        
        if len(pca.explained_variance_ratio_) >= 2:

            figure = "PCA_Artificial_Train_Model_" + str(k)
            title = "F_" + str(FEATURE_DIM) + "_K_"+ str(KEY_DIM) + "_wf_" + str(w_f) + "_Model_" + str(k)
            plt.figure()
            print(len(valid_feature[:,0]))
            plt_file = plot_path + str(extra) + "_" + str(figure) + ".png"
            plt.scatter(valid_feature[:,0], valid_feature[:,1])
            plt.grid()
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.title(title)
            plt.savefig(plt_file)
            plt.show()
            plt.close('all')
            
        if FEATURE_DIM == 32:
            matrix_value_csv = plot_path + str(extra) + "_value_matrix_Model_" + str(k) + ".csv"
            matrix_value = []
            for n,p in net_list[k].named_parameters():
                if n == "value_matrix":
                    for i in range(KEY_DIM):
                        p_i = list(p[i].detach().numpy())
                        sortp = list(np.argsort(p_i))
                        matrix_value.append(p_i)
                        sort_index = []
                        for j in range(len(p_i)):
                            sort_index.append(int(sortp.index(j)))
                        matrix_value.append(sort_index)
                    matrix_value_df = pd.DataFrame(np.array(matrix_value).transpose())
                    matrix_value_df.to_csv(matrix_value_csv)

    return

def run(input_csv,
        output_csv,
        weight_csv = "",
        mode = "NORMAL",
        username = "Artificial",
        ARTIFICIAL = False,
        K_FOLDER = 5,
        DATE = DATE,
        MOMENTUM = 0.9,
        EPOCH = 1000,
        BATCH_NORM = False,
        K = 6,
        F = 10,
        Q = 9,
        REG = L0,
        LEARNING_RATE = 0.05,
        WEIGHT_DECAY = 0.005,
        LOSS = CEL,
        ACT = SIGMOID,
        BATCH_SIZE = 10,
        OPTIMIZER = SGD,
        NET = ATTENTION,
        w_f = "FIXED",
        w_f_type = "Eye",
        VALIDATE_NUMBER = 1000,
        WD = "0005",
        extra_msg = "",
        ATTENTION_REG = False,
        ATTENTION_REG_WEIGHT = 0.05,
        ):

    WEIGHT_DECAY = torch.tensor(WEIGHT_DECAY).float()

    if mode == "NORMAL":
        run_normal(input_csv = input_csv,
        output_csv = output_csv,
        weight_csv = weight_csv,
        username = username,
        ARTIFICIAL = ARTIFICIAL,
        MOMENTUM = MOMENTUM,
        BATCH_NORM = BATCH_NORM,
        DATE = DATE,
        EPOCH = EPOCH,
        KEY_DIM = K,
        FEATURE_DIM = F,
        QUERY_DIM = Q,
        REG = REG,
        LEARNING_RATE = LEARNING_RATE,
        WEIGHT_DECAY = WEIGHT_DECAY,
        LOSS = LOSS,
        ACT = ACT,
        BATCH_SIZE = BATCH_SIZE,
        OPTIMIZER = OPTIMIZER,
        NET = NET,
        w_f = w_f,
        w_f_type = w_f_type,
        WD = WD,
        extra_msg = extra_msg
        )
    elif mode == "CV":
        run_cross_validation(input_csv = input_csv,
        output_csv = output_csv,
        weight_csv = weight_csv,
        username = username,
        ARTIFICIAL = ARTIFICIAL,
        BATCH_NORM = BATCH_NORM,
        K_FOLDER = K_FOLDER,
        MOMENTUM = MOMENTUM,
        DATE = DATE,
        EPOCH = EPOCH,
        KEY_DIM = K,
        FEATURE_DIM = F,
        QUERY_DIM = Q,
        REG = REG,
        LEARNING_RATE = LEARNING_RATE,
        WEIGHT_DECAY = WEIGHT_DECAY,
        ATTENTION_REG = ATTENTION_REG,
        ATTENTION_REG_WEIGHT = ATTENTION_REG_WEIGHT,
        LOSS = LOSS,
        ACT = ACT,
        BATCH_SIZE = BATCH_SIZE,
        OPTIMIZER = OPTIMIZER,
        NET = NET,
        w_f = w_f,
        w_f_type = w_f_type,
        VALIDATE_NUMBER = VALIDATE_NUMBER,
        WD = WD,
        extra_msg = extra_msg
        )

    ##################
    



