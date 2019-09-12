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

#####################################
class Attention_Net(nn.Module):
    def __init__(self, dataset, params = (5,10,8), activation = "sigmoid"):
        super(Attention_Net,self).__init__()
        self.dataset = dataset
        self.input_dim = self.dataset.input_dim
        self.output_dim = self.dataset.output_dim
        self.condition_dim = self.input_dim - self.output_dim

        self.query_dim = int(params[0])
        self.key_dim = int(params[1])
        self.feature_dim = int(params[2])
        self.linear_layer1 = nn.Linear(self.input_dim, self.query_dim)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU(True)

        self.key_matrix = torch.nn.Parameter(torch.randn(self.query_dim, self.key_dim))
        self.value_matrix = torch.nn.Parameter(torch.randn(self.key_dim, self.feature_dim))
        self.linear_layer2 = nn.Linear(self.feature_dim, self.output_dim)

        init.xavier_uniform(self.linear_layer1.weight)
        init.xavier_uniform(self.linear_layer2.weight)
        init.normal(self.linear_layer1.bias, mean = 0, std = 1)
        init.normal(self.linear_layer2.bias, mean = 0, std = 1)

    def forward(self,x,mode = "train"):
        #Encoder
        inp = x
        zero_mask = self.get_zero_mask(x)
        x = self.linear_layer1(x)
        x = self.act(x)
        x = x.mm(self.key_matrix)
        x = F.softmax(x,dim = 1)
        self.distribute = x
        x = x.mm(self.value_matrix)

        #Decoder
        x = self.linear_layer2(x)
        inf_mask = self.get_inf_mask(inp,x)
        if mode=="valid":
            x = x.mul(inf_mask)
        x = F.softmax(x, dim = 1)
        if mode=="train":
            x = x.mul(zero_mask)
        return x,self.distribute

    def get_inf_mask(self,inp,x):
        inf = float('inf')
        x_shape = x.shape
        #print(x_shape[0])
        #print(x_shape[1])
        mask = []
        inp = inp.data.numpy()
        x = x.data.numpy()
        for i in range(x_shape[0]):
            sub_mask = []
            for j in range(x_shape[1]):
                if inp[i][self.condition_dim + j] == 0:
                    if x[i][j] > 0:
                        sub_mask.append(-inf)
                    else:
                        sub_mask.append(inf)
                else:
                    sub_mask.append(1)
            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        #print(mask.shape)
        #print(mask)
        return mask

    def get_zero_mask(self,x):
        x = x.data.numpy()
        mask = []
        for batch in x:
            sub_mask = []
            for i in range(self.output_dim):
                if batch[self.condition_dim + i] == 0:
                    sub_mask.append(0)
                else:
                    sub_mask.append(1)
            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        return mask


class Linear_Net(nn.Module):
    def __init__(self, dataset,params = 8, activation = "sigmoid"):
        super(Linear_Net,self).__init__()
        self.dataset = dataset
        self.input_dim = self.dataset.input_dim
        self.output_dim = self.dataset.output_dim
        self.condition_dim = self.input_dim - self.output_dim

        self.feature_dim = int(params)
        self.linear_layer1 = nn.Linear(self.input_dim, self.feature_dim)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU(True)

        
        self.linear_layer2 = nn.Linear(self.feature_dim, self.output_dim)

        init.xavier_uniform(self.linear_layer1.weight)
        init.xavier_uniform(self.linear_layer2.weight)
        init.normal(self.linear_layer1.bias, mean = 0, std = 1)
        init.normal(self.linear_layer2.bias, mean = 0, std = 1)

    def forward(self,x,mode = "train"):
        #Encoder
        inp = x
        zero_mask = self.get_zero_mask(x)
        x = self.linear_layer1(x)
        x = self.act(x)
        #Decoder
        x = self.linear_layer2(x)
        inf_mask = self.get_inf_mask(inp,x)
        if mode=="valid":
            x = x.mul(inf_mask)
        x = F.softmax(x, dim = 1)
        if mode=="train":
            x = x.mul(zero_mask)
        return x

    def get_inf_mask(self,inp,x):
        inf = float('inf')
        x_shape = x.shape
        #print(x_shape[0])
        #print(x_shape[1])
        mask = []
        inp = inp.data.numpy()
        x = x.data.numpy()
        for i in range(x_shape[0]):
            sub_mask = []
            for j in range(x_shape[1]):
                if inp[i][self.condition_dim + j] == 0:
                    if x[i][j] > 0:
                        sub_mask.append(-inf)
                    else:
                        sub_mask.append(inf)
                else:
                    sub_mask.append(1)
            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        #print(mask.shape)
        #print(mask)
        return mask

    def get_zero_mask(self,x):
        x = x.data.numpy()
        mask = []
        for batch in x:
            sub_mask = []
            for i in range(self.output_dim):
                if batch[self.condition_dim + i] == 0:
                    sub_mask.append(0)
                else:
                    sub_mask.append(1)
            mask.append(sub_mask)
        mask = torch.from_numpy(np.array(mask)).float()
        return mask
