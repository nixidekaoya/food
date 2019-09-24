#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import pandas as pd
import numpy as np
import math
import os
import random

csv_file = "/home/li/datasets/csv/20190922_limofei_1000_food.csv"
input_csv = "/home/li/food/data/20190922_limofei_1000_input.csv"
output_csv = "/home/li/food/data/20190922_limofei_1000_output.csv"
list_csv_path = "/home/li/webapi/domain/combine_lists.csv"


df = pd.read_csv(csv_file)
list_csv = pd.read_csv(list_csv_path)
A_list = list(list_csv.A)
B_list = list(list_csv.B)
A_list_len = len(A_list)
B_list_len = len(B_list)
AB_list = A_list + B_list
input_dim = len(AB_list)
output_dim = len(B_list)
index = list(df.index)

print(df.shape)
data_num = df.shape[0]

input_matrix = []
output_matrix = []

for index,r in df.iterrows():
    input_array = np.zeros(input_dim)
    input_array[int(r["AListID"])] = 1
    input_array[A_list_len + int(r["BListID1"])] = 1
    input_array[A_list_len + int(r["BListID2"])] = 1
    input_array[A_list_len + int(r["BListID3"])] = 1
    input_array[A_list_len + int(r["BListID4"])] = 1
    input_matrix.append(input_array)

    output_array = np.zeros(output_dim)
    output_array[int(r["BListSelectID"])] = 1
    output_matrix.append(output_array)
    print(sum(input_array))
    

print(np.array(input_matrix).shape)
print(np.array(output_matrix).shape)
data_f_input = pd.DataFrame(input_matrix, columns = AB_list, index = range(data_num))
data_f_input.to_csv(input_csv)
data_f_output = pd.DataFrame(output_matrix, columns = B_list, index = range(data_num))
data_f_output.to_csv(output_csv)
