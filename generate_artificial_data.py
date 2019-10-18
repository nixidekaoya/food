#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import math
import random
import itertools
import matplotlib.pyplot as plt
from numpy.random import choice


#PARAMS
class_number = 2
item_number = 32
data_number = 100000
choice_number = 4
total_number = class_number * item_number
DATE = "20191016"
artificial_path = "/home/li/food/artificial_data/"
extra = "_ITEM_NO_" + str(item_number) + "_CLASS_NO_" + str(class_number)
log_file_path = artificial_path + str(DATE) + str(extra) + ".txt"

input_csv = artificial_path + "Artificial_" + str(DATE) + str(extra) + "_DATA_NO_" + str(data_number) + "_CHOICE_NO_" + str(choice_number) + "input.csv"
output_csv = artificial_path + "Artificial_" + str(DATE) + str(extra) + "_DATA_NO_" + str(data_number) + "_CHOICE_NO_" + str(choice_number) + "output.csv"

weight_csv = artificial_path + "Artificial_" + str(DATE) + str(extra) + "_DATA_NO_" + str(data_number) + "_CHOICE_NO_" + str(choice_number) + "weights.csv"


weights = []
for j in range(class_number):
    weights.append([])
    for i in range(item_number):
        weights[j].append(np.random.rand())
    weight_sum = sum(weights[j])
    for i in range(item_number):
        weights[j][i] /= weight_sum

weights.append([])
for i in range(item_number):
    weights[class_number].append(sum([weights[j][i] for j in range(class_number)]))

for j in range(class_number+1):
    sorts = list(np.argsort(weights[j]))
    weights_index = []
    for i in range(item_number):
        weights_index.append(sorts.index(i))
    weights.append(weights_index)



with open(log_file_path,'w') as log_file:
    for j in range(class_number):
        log_file.write("Class No." + str(i) + "\r\n")
        for i in range(item_number):
            log_file.write(str(weights[j][i]) + "\r\n")

print(weights)

input_matrix = []
output_matrix = []

for i in range(data_number):
    ### INPUT
    input_array = []
    choice_rand_list = random.sample(range(item_number),int(choice_number))
    class_no = random.choice(range(class_number))
    
    
    if class_no == 0:
        input_array.append(0)
    elif class_no == 1:
        input_array.append(1)
    
    for j in range(item_number):
        if j in choice_rand_list:
            input_array.append(1)
        else:
            input_array.append(0)
            

    input_matrix.append(input_array)
    
    weights_list = []
    for j in range(choice_number):
        weights_list.append(weights[class_no][choice_rand_list[j]])
    sum_weights = sum(weights_list)
    for j in range(choice_number):
        weights_list[j] /= sum_weights
    selected_choice = choice_rand_list[np.argmax(weights_list)]
    #print(choice_rand_list)
    #print(weights_list)
    #print(selected_choice)
    output_array = np.zeros(item_number).tolist()
    output_array[selected_choice] = 1
    output_matrix.append(output_array)
    
    

input_columns = ['flag']
input_columns.extend(list(range(item_number)))

print(input_columns)
weight_columns = ["table1","table2","sum_table","rank1","rank2","rank_sum12"]

data_f_input = pd.DataFrame(input_matrix, columns = input_columns, index = range(data_number))
print(data_f_input.shape)
data_f_input.to_csv(input_csv)
data_f_output = pd.DataFrame(output_matrix, columns = range(item_number), index = range(data_number))
print(data_f_output.shape)
data_f_output.to_csv(output_csv)
data_f_weights = pd.DataFrame(np.array(weights).transpose(), columns = weight_columns, index = range(item_number))
print(data_f_weights.shape)
data_f_weights.to_csv(weight_csv)


