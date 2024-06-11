import os
import math
import copy
from itertools import chain
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import config
from utils import *

conf = config.config()
num_classes = conf.num_classes
batchsize = conf.batchsize
datafile = conf.datafile
device = conf.device_detection()
T = conf.T
lamda = 0.8151841430223141
m = nn.Softmax(dim=1)
# ======================================================================================================================
# data loading
# ======================================================================================================================
transform=transforms.Compose([
                            ToTensor(),
                            transforms.Normalize(mean=[0.456],std=[0.224]),
                            ])
testset=myDataset(datafile+"test/", transform=transform)
test_load=DataLoader(testset, batch_size=batchsize, shuffle=False)
# ======================================================================================================================
# model set
# ======================================================================================================================
fin_model = []
#model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/resnet.h5') #,map_location='cpu')
#model = copy.deepcopy(model)
#fin_model.append(model)
#model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/densenet.h5') #,map_location='cpu')
#model = copy.deepcopy(model)
#fin_model.append(model)
model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/1.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/2.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/3.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/4.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/ljh/code_2D_DBT900/ENAS/modeltest/5.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
print(model)
t
# ======================================================================================================================
# test single model
# ======================================================================================================================
print('Performance of each model')
pareto = np.zeros((len(fin_model),2))
AUC = np.zeros((len(fin_model), 1))
for i in range(len(fin_model)):
    model = fin_model[i]
    model = model.to(device)
    model.eval() 
    index = 0
    test_preds = torch.zeros(len(testset), 1)
    test_probs = torch.zeros(len(testset), 1)
    test_labels = torch.zeros(len(testset), 1)
    with torch.no_grad():
        for X, y, in test_load:
            X, y = Variable(X.to(device)), Variable(y.to(device))
            pred = model(X)
            probs = m(pred)
            _, pred_y = torch.max(probs, 1)
            test_labels[index:index + len(y), 0] = y
            test_probs[index:index + len(y), 0] = probs[:, 1]
            test_preds[index:index + len(y), 0] = pred_y
            index += len(y)
    test_metric = compute_measures(test_labels, test_preds, test_probs)
    metrics_print(test_metric)
    print()
    pareto[i][0] = test_metric['sen']
    pareto[i][1] = test_metric['spe']
    AUC[i] = test_metric['auc']
# ======================================================================================================================
# ERE
# ======================================================================================================================
#fin_model.pop(0)
#fin_model.pop(0)
#pareto = np.delete(pareto, [0,1], axis=0) 
#AUC = np.delete(AUC, [0,1], axis=0)
w = cal_weight(pareto, AUC, lamda)
print(w)
test_labels, test_preds, test_probs, test_uncertainty = ERE(fin_model, w, testset, device, num_classes, T)
test_metric = compute_measures(test_labels, test_preds, test_probs)
metrics_print(test_metric)
print()
# ======================================================================================================================
# ERE:uncertainty_sort + compute_measures
# ======================================================================================================================
uncertainty_sort, indices = torch.sort(torch.flatten(test_uncertainty), descending=True, dim=-1)  
print(uncertainty_sort[0])
length = len(test_uncertainty)


num = 0
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length/4)]:
        num += 1
index = 0
test_preds2 = torch.zeros(num, 1)
test_probs2 = torch.zeros(num, 1)
test_u2 = torch.zeros(num, 1)
test_labels2 = torch.zeros(num, 1)
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length/4)]:
        test_preds2[index:index + 1, 0] = test_preds[i]
        test_probs2[index:index + 1, 0] = test_probs[i]
        test_u2[index:index + 1, 0] = test_uncertainty[i]
        test_labels2[index:index + 1, 0] = test_labels[i]
        index += 1
test_metric = compute_measures(test_labels2, test_preds2, test_probs2)
metrics_print(test_metric)
print(uncertainty_sort[int(length/4)])
print('\t')


num = 0
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length/2)]:
        num += 1
index = 0
test_preds2 = torch.zeros(num, 1)
test_probs2 = torch.zeros(num, 1)
test_u2 = torch.zeros(num, 1)
test_labels2 = torch.zeros(num, 1)
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length/2)]:
        test_preds2[index:index + 1, 0] = test_preds[i]
        test_probs2[index:index + 1, 0] = test_probs[i]
        test_u2[index:index + 1, 0] = test_uncertainty[i]
        test_labels2[index:index + 1, 0] = test_labels[i]
        index += 1
test_metric = compute_measures(test_labels2, test_preds2, test_probs2)
metrics_print(test_metric)
print(uncertainty_sort[int(length/2)])
print('\t')


num = 0
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length*3/4)]:
        num += 1
index = 0
test_preds2 = torch.zeros(num, 1)
test_probs2 = torch.zeros(num, 1)
test_u2 = torch.zeros(num, 1)
test_labels2 = torch.zeros(num, 1)
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length*3/4)]:
        test_preds2[index:index + 1, 0] = test_preds[i]
        test_probs2[index:index + 1, 0] = test_probs[i]
        test_u2[index:index + 1, 0] = test_uncertainty[i]
        test_labels2[index:index + 1, 0] = test_labels[i]
        index += 1
test_metric = compute_measures(test_labels2, test_preds2, test_probs2)
metrics_print(test_metric)
print(uncertainty_sort[int(length*3/4)])
print()


