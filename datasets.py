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

###########################
class FoodDataset(Dataset):
    def __init__(self,input_csv,output_csv):
        self.input_data = pd.read_csv(input_csv)
        self.output_data = pd.read_csv(output_csv)
        #print(self.input_data.shape)
        #print(self.output_data.shape)
        self.B_list = list(self.output_data.columns)
        self.B_list_len = len(self.B_list) - 1
        self.output_dim = self.B_list_len
        print(self.output_dim)
        self.AB_list = list(self.input_data.columns)
        self.AB_list_len = len(self.AB_list) - 1
        self.input_dim = self.AB_list_len
        print(self.input_dim)
        self.A_list_len = self.AB_list_len - self.B_list_len
        self.A_list = self.AB_list[:self.A_list_len]

        self.input_list = []
        self.output_list = []
        

        for index,row in self.input_data.iterrows():
            self.input_list.append(list(row)[1:])

        for index,row in self.output_data.iterrows():
            self.output_list.append(list(row)[1:])

        #print(len(self.input_list[1]))
        #print(len(self.output_list[1]))
        self.data_num = len(self.input_list)

    def __getitem__(self,index):
        input_item = torch.from_numpy(np.array(self.input_list[index])).float()
        output_item = torch.from_numpy(np.array(self.output_list[index])).long()

        return input_item,output_item

    def __len__(self):
        return self.data_num
        
