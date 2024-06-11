#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import config
from utils import *

conf = config.config()
save_path = conf.save_path
it_num = conf.it_num
pop_num = conf.pop_num
CS = conf.CS
num_classes = conf.num_classes
batchsize = conf.batchsize
epoch_num = conf.epoch_num
LR = conf.LR
best_loss = conf.best_loss
early_stop = conf.early_stop
max_early_stop = conf.max_early_stop
modelsavefile = conf.modelsavefile
lossfile = conf.lossfile
datafile = conf.datafile
device = conf.device_detection()
T = conf.T

def objective(space): 
    ftxt = open(save_path, 'a')
    lamda = space['lamda']
    mut_rate = space['mut_rate']
    min_model = space['min_model']
    ftxt.write('lamda=' + str(lamda))
    ftxt.write(' ' + 'mut_rate=' + str(mut_rate))
    ftxt.write(' ' + 'min_model=' + str(min_model))
    # ======================================================================================================================
    # data loading
    # ======================================================================================================================
    transform=transforms.Compose([
                                ToTensor(),
                                transforms.Normalize(mean=[0.456],std=[0.224]),
                                ])
    trainset=myDataset(datafile+"train/", transform=transform)
    train_load=DataLoader(trainset, batch_size=batchsize, shuffle=True)
    valset=myDataset(datafile+"val/", transform=transform)
    val_load=DataLoader(valset, batch_size=batchsize, shuffle=False)

    # ======================================================================================================================
    # optimization
    # ======================================================================================================================
    EPOP = ini_pop()
    Epa, f_obj, AUC_f = computing_fitnesss_ini(0, EPOP, train_load, val_load, epoch_num, LR, trainset, valset, device, num_classes, lossfile, best_loss, early_stop, max_early_stop) 
    save_pa = []
    save_pa.append(Epa)
    for it in range(0, it_num): 
        cloneover = Clonef(copy.deepcopy(EPOP), Epa, CS)
        mut_over = PNmutation(copy.deepcopy(cloneover), mut_rate)
        NPOP = []
        NPOP.extend(copy.deepcopy(EPOP))
        NPOP.extend(copy.deepcopy(mut_over))
        NPOP = delete(copy.deepcopy(NPOP))
        while len(NPOP) < pop_num: 
            Npa, f_obj, AUC_f = computing_fitnesss_ini(it, NPOP, train_load, val_load, epoch_num, LR, trainset, valset, device, num_classes, lossfile, best_loss, early_stop, max_early_stop) 
            cloneover = Clonef(copy.deepcopy(NPOP), Npa, CS)
            mut_over = PNmutation(copy.deepcopy(cloneover), mut_rate)
            NNPOP = []
            NNPOP.extend(copy.deepcopy(NPOP))
            NNPOP.extend(copy.deepcopy(mut_over))
            NNPOP = delete(copy.deepcopy(NNPOP))
            NPOP = copy.deepcopy(NNPOP)
        Npa, f_obj, AUC_f, f_model= computing_fitness(it, copy.deepcopy(NPOP), train_load, val_load, epoch_num, LR, trainset, valset, device, num_classes, lossfile, best_loss, early_stop, max_early_stop)
        FL = FNDSF(Npa)
        FL = np.array(FL)
        EPOP, Epal, E_f, E_AUC, E_model = UPPNSGAIIf(copy.deepcopy(NPOP), Npa, FL, pop_num, f_obj, AUC_f, f_model) 
        Epa = Epal[:,0:2]
        save_pa.append(Epa)
    #plot_per_it(lossfile, save_pa, 'optimization.png')
    # ======================================================================================================================
    # select models from pareto population
    # ======================================================================================================================
    pareto, FL, fin_POP, fin_f, fin_AUC, fin_model = select_models(EPOP, Epal, E_f, E_AUC, E_model, min_model)
    for i in range(0, len(fin_model)):
        torch.save(fin_model[i], modelsavefile+time.strftime("%Y-%m-%d %X", time.localtime())+'_'+str(i)+'_model.h5')
    # ======================================================================================================================
    # ERE
    # ======================================================================================================================
    w = cal_weight(pareto, fin_AUC, lamda)
    print(pareto, FL, fin_f, fin_AUC)
    val_labels, val_preds, val_probs, val_uncertainty = ERE(fin_model, w, valset, device, num_classes, T)
    val_metric = compute_measures(val_labels, val_preds, val_probs)
    metrics_print(val_metric)
    # ======================================================================================================================
    # save result
    # ======================================================================================================================
    SEN = round(val_metric['sen'], 4)
    SPE = round(val_metric['spe'], 4)
    AUC = round(val_metric['auc'], 4)
    ACC = round(val_metric['acc'], 4)
    SSEN = ' ' + 'SEN=' + str(SEN)
    SSPE = ' ' + 'SPE=' + str(SPE)
    SAUC = ' ' + 'AUC=' + str(AUC)
    SACC = ' ' + 'ACC=' + str(ACC) + '\n'
    ftxt.write(SSEN)
    ftxt.write(SSPE)
    ftxt.write(SAUC)
    ftxt.write(SACC)
    ftxt.write(' ' + 'POP=' + '\n' + str(fin_POP) + '\n')
    ftxt.write(' ' + 'pareto=' + '\n' + str(pareto) + '\n')
    ftxt.write(' ' + 'FL=' + '\n' + str(FL) + '\n')
    ftxt.write(' ' + 'auc=' + '\n' + str(fin_AUC) + '\n')
    ftxt.write(' ' + 'acc=' + '\n' + str(fin_f) + '\n')
    ftxt.write(' ' + '权重=' + '\n' + str(w) + '\n')
    ftxt.write(' ' + 'pop_num=' + str(pop_num))
    ftxt.write(' ' + 'it_num=' + str(it_num))
    ftxt.write(' ' + 'CS=' + str(CS))
    ftxt.write(' ' + 'num_classes=' + str(num_classes))
    ftxt.write(' ' + 'batchsize=' + str(batchsize))
    ftxt.write(' ' + 'epoch_num=' + str(epoch_num))
    ftxt.write(' ' + 'LR=' + str(LR))
    ftxt.write(' ' + 'max_early_stop=' + str(max_early_stop) + '\n')
    ftxt.write('\n')
    ftxt.close()
    f = 1 - AUC
    return f
