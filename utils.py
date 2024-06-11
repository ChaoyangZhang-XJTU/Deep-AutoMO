import os
import sys
import copy
import math
import random
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable 
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt
from EvoCNNModel import *
import config

conf = config.config()

def ini_pop(): # initialize population
    EPOP = []
    set = ModelSettings()
    for _ in range(0,conf.pop_num):
        individual = []
        # initialize how many resnet unit/pooling unit/densenet unit will be used
        num_resnet = np.random.randint(set.min_units, set.max_units+1)
        num_pool = np.random.randint(set.min_units, set.max_units+1)
        num_densenet = np.random.randint(set.min_units, set.max_units+1)
        # the types of units
        total_length = num_resnet + num_pool + num_densenet
        units = np.zeros(total_length, np.int32)
        units[0:num_resnet] = 1 #resnet
        units[num_resnet:num_resnet+num_pool] = 2 #pool
        units[num_resnet+num_pool:num_resnet+num_pool+num_densenet] = 3 #densenet
        for _ in range(10):
            np.random.shuffle(units)
        while units[0] == 2: # pooling should not be the first unit
            np.random.shuffle(units)
        # initialize each unit       
        in_channel = 3
        for j in range(0,len(units)):
            layer = []
            if units[j] == 1:
                amount = np.random.randint(set.min_amount, set.max_amount1+1)
                out_channel = set.output_channles[np.random.randint(0, len(set.output_channles))]
                layer.append(amount)
                layer.append(in_channel)
                layer.append(out_channel)
                in_channel = out_channel
            if units[j] == 2:
                pool_type = set.pool_types[np.random.randint(0, len(set.pool_types))]
                layer.append(pool_type)
            if units[j] == 3:
                k = set.k_list[np.random.randint(0, len(set.k_list))]
                if k == 12:
                    amount = np.random.randint(set.min_amount, set.max_amount1+1)
                    max_input_channel = set.max_input_channels[0]
                if k == 20:
                    amount = np.random.randint(set.min_amount, set.max_amount1+1)
                    max_input_channel = set.max_input_channels[1]
                if k == 40:
                    amount = np.random.randint(set.min_amount, set.max_amount2+1)
                    max_input_channel = set.max_input_channels[2]
                layer.append(k)
                layer.append(amount)
                layer.append(in_channel)
                layer.append(max_input_channel)
                true_input_channel = in_channel
                if true_input_channel > max_input_channel:
                    true_input_channel = max_input_channel
                in_channel = true_input_channel + k * amount

            individual.append(layer)

        EPOP.append(individual)
    return EPOP


def computing_fitnesss_ini(it0, NPOP, train_load, val_load, epoch_num, LR, trainset, valset, device, num_classes, lossfile, best_loss1, early_stop1, max_early_stop1):
    # ======================================================================================================================
    # device detection
    # ======================================================================================================================
    device = device
    print('device: ', device)

    f = np.zeros((len(NPOP), 2))
    obj_f = np.zeros((len(NPOP), 1))
    AUC_f = np.zeros((len(NPOP), 1))
    for t in range(0, len(NPOP)):
        # ======================================================================================================================
        # model building
        # ======================================================================================================================
        model = EvoCNNModel(num_classes, NPOP[t])
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss() 
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=5*1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1) # lr = lr *gamma
        m = nn.Softmax(dim=1)

        train_labels = Variable(torch.zeros(len(trainset), 1).to(device))
        train_preds = Variable(torch.zeros(len(trainset), 1).to(device))
        train_probs = Variable(torch.zeros(len(trainset), 1).to(device))
        train_losses = Variable(torch.zeros(epoch_num, 1).to(device))
        train_metrics = []

        val_labels = Variable(torch.zeros(len(valset), 1).to(device))
        val_preds = Variable(torch.zeros(len(valset), 1).to(device))
        val_probs = Variable(torch.zeros(len(valset), 1).to(device))
        val_losses = Variable(torch.zeros(epoch_num, 1).to(device))
        val_metrics = []

        model_all = []

        # ======================================================================================================================
        # model training, validation, testing
        # ======================================================================================================================
        best_loss = best_loss1
        early_stop = early_stop1
        max_early_stop = max_early_stop1
        
        print("\033[1;32m    ini [{}] \033[0m".format(it0))
        for i in range(1, epoch_num + 1):
            print('\n')
            print("\033[1;32m    Epoch [{}/{}]     ini [{}]\033[0m".format(i, epoch_num, it0))
            print("-" * 45)

            # model training
            model.train()
            train_loss = 0.0
            cprint('r', "model is training...")
            index = 0
            for X, y, in train_load:
                X, y = Variable(X.to(device)), Variable(y.to(device)) # to device
                pred = model(X)                     # prediction, logits
                probs = m(pred)                     # probability
                _, pred_y = torch.max(probs, 1)     # predicted label
                optimizer.zero_grad()
                loss = loss_fn(pred, y)
                loss.backward()                     # backward pass
                optimizer.step()                    # weights update
                train_loss += len(y) * loss.item()  
                train_labels[index:index+len(y), 0] = y
                train_preds[index:index+len(y), 0] = pred_y
                train_probs[index:index+len(y), 0] = probs[:,1]

                index += len(y)

            scheduler.step()  

            print('train loss :', round(1.0*train_loss/len(trainset), 4))
            train_metric = compute_measures(train_labels, train_preds, train_probs)
            train_losses[i - 1, 0] = round(1.0 * train_loss / len(trainset), 4)
            train_metrics.append(train_metric)
            metrics_print(train_metric)
            
            # model validating
            model.eval() 
            val_loss = 0.0
            cprint('r', "\nmodel is validating...")
            index = 0
            with torch.no_grad():
                for X, y, in val_load:
                    X, y = Variable(X.to(device)), Variable(y.to(device))
                    pred = model(X)
                    probs = m(pred)
                    _, pred_y = torch.max(probs, 1)
                    loss = loss_fn(pred, y)
                    val_loss += len(y) * loss.item()
                    #import pdb
                    #pdb.set_trace()
                    val_labels[index:index + len(y), 0] = y
                    val_preds[index:index + len(y), 0] = pred_y
                    val_probs[index:index + len(y), 0] = probs[:, 1]

                    index += len(y)
            print('val loss :', round(1.0 * val_loss / len(valset), 4))
            val_metric = compute_measures(val_labels, val_preds, val_probs)
            val_metrics.append(val_metric)
            val_losses[i - 1, 0] = round(1.0 * val_loss / len(valset), 4)
            metrics_print(val_metric)

            if val_losses[i - 1, 0] < best_loss:
                best_loss = val_losses[i - 1, 0]
                early_stop = 0
                # model saving
                tt = copy.deepcopy(model) 
                model_all.append(tt)
            else:
                early_stop += 1
            if early_stop == max_early_stop:
                print("Early Stopping")
                break
        plot_per_epoch(lossfile, val_losses.cpu().numpy(), 'Valid loss.png', train_losses.cpu().numpy())
        
        f[t][0] = val_metrics[len(model_all)-1]['sen']
        f[t][1] = val_metrics[len(model_all)-1]['spe']
        obj_f[t] = val_metrics[len(model_all)-1]['acc']
        AUC_f[t] = val_metrics[len(model_all)-1]['auc']

    return f, obj_f, AUC_f

def Clonef(POP, pa, CS): # clone
    N = len(POP)
    POP, pa, padis = CDAf1(copy.deepcopy(POP), pa) # calculate crowding-distancein
    aa = [k for k, x in enumerate(padis) if x == float("inf")]
    bb = [k for k, x in enumerate(padis) if x != float("inf")]
    NC=np.zeros((len(POP),1))
    if len(bb) > 0:
        padis[aa] = 2 * max(padis[bb])
        if sum(padis)==0:
            for i in range(0,len(padis)):
                padis[i]=1/len(padis)
            NC = np.ceil(CS * padis)
        else:
            NC = np.ceil(CS * padis / sum(padis))
    else:
        NC = np.ceil(CS / len(aa)) + np.zeros([N, 1])
    NPOP = []
    for i in range(0, N):
        print(NC)
        if NC[i] == 0:
            continue
        else:
            for _ in range(int(NC[i])):
                NPOP.append(copy.deepcopy(POP[i]))
    return NPOP

def PNmutation(POP, mut_rate):
    clone0ver = []
    mutation_list = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    for i in range(len(POP)):
        indi = POP[i]
        rate = random.random()
        if rate < mut_rate:
            mutation_type = mutation_list[np.random.randint(0, len(mutation_list))]
            if len(indi) > 1:
                mutation_position = np.random.randint(1, len(indi)) # determine the position where a unit would be mutated
                if mutation_type == 0:
                    indi = do_add_unit_mutation(indi, mutation_position)
                elif mutation_type == 1:
                    indi = do_remove_unit_mutation(indi, mutation_position)
                elif mutation_type == 2:
                    indi = do_alter_mutation(indi, mutation_position)
        clone0ver.append(indi)
    return clone0ver

def do_add_unit_mutation(indi, mutation_position):
    set = ModelSettings()
    layer = []
    # determine the unit type for adding
    u_ = random.random()
    if u_ < 0.333:
        type_ = 1
    elif u_ < 0.666:
        type_ = 2
    else:
        type_ = 3
    if type_ == 2:
        num_exist_pool_units = 0
        for i in range(len(indi)):
            if len(indi[i]) == 1:
                num_exist_pool_units += 1
        if num_exist_pool_units > set.max_units:
            u_ = random.random()
            type_ = 1 if u_ < 0.5 else 3

    #do the details
    if type_ == 2:
        pool_type = set.pool_types[np.random.randint(0, len(set.pool_types))]
        layer.append(pool_type)
    else:
        for i in range(mutation_position-1, -1, -1):
            if len(indi[i]) == 3:
                _in_channel = indi[i][2]
                break
            if len(indi[i]) == 4:
                true_input_channel = indi[i][2]
                if true_input_channel > indi[i][3]:
                    true_input_channel = indi[i][3]
                _in_channel = true_input_channel + indi[i][0] * indi[i][1]
                break
        
        if type_ == 1:
            amount = np.random.randint(set.min_amount, set.max_amount1+1)
            out_channel = set.output_channles[np.random.randint(0, len(set.output_channles))]
            layer.append(amount)
            layer.append(_in_channel)
            layer.append(out_channel)
            next_in_channel = out_channel
        if type_ == 3:
            k = set.k_list[np.random.randint(0, len(set.k_list))]
            if k == 12:
                amount = np.random.randint(set.min_amount, set.max_amount1+1)
                max_input_channel = set.max_input_channels[0]
            if k == 20:
                amount = np.random.randint(set.min_amount, set.max_amount1+1)
                max_input_channel = set.max_input_channels[1]
            if k == 40:
                amount = np.random.randint(set.min_amount, set.max_amount2+1)
                max_input_channel = set.max_input_channels[2]
            layer.append(k)
            layer.append(amount)
            layer.append(_in_channel)
            layer.append(max_input_channel)
            true_input_channel = _in_channel
            if true_input_channel > max_input_channel:
                true_input_channel = max_input_channel
            next_in_channel = true_input_channel + k * amount

        for i in range(mutation_position, len(indi)):
            if len(indi[i]) == 3:
                indi[i][1] = next_in_channel
                break
            if len(indi[i]) == 4:
                true_input_channel = indi[i][2]
                if true_input_channel > indi[i][3]:
                    true_input_channel = indi[i][3]
                true_out_channel_channel = true_input_channel + indi[i][0] * indi[i][1]
                
                indi[i][2] = next_in_channel
                true_input_channel = indi[i][2]
                if true_input_channel > indi[i][3]:
                    true_input_channel = indi[i][3]
                estimated_out_channel = true_input_channel + indi[i][0] * indi[i][1]

                if estimated_out_channel == true_out_channel_channel:
                    break
                else:
                    next_in_channel = estimated_out_channel
    indi.insert(mutation_position, layer)
    return indi

def do_remove_unit_mutation(indi, mutation_position):
    if len(indi) > 1:
        if len(indi[mutation_position]) == 3 or len(indi[mutation_position]) == 4:
            for i in range(mutation_position-1, -1, -1):
                if len(indi[i]) == 3:
                    _in_channel = indi[i][2]
                    break
                if len(indi[i]) == 4:
                    true_input_channel = indi[i][2]
                    if true_input_channel > indi[i][3]:
                        true_input_channel = indi[i][3]
                    _in_channel = true_input_channel + indi[i][0] * indi[i][1]
                    break
            if mutation_position != (len(indi)-1):
                for i in range(mutation_position+1, len(indi)):
                    if len(indi[i]) == 3:
                        indi[i][1] = _in_channel
                        break
                    if len(indi[i]) == 4:
                        true_input_channel = indi[i][2]
                        if true_input_channel > indi[i][3]:
                            true_input_channel = indi[i][3]
                        true_out_channel_channel = true_input_channel + indi[i][0] * indi[i][1]
                        
                        indi[i][2] = _in_channel
                        true_input_channel = indi[i][2]
                        if true_input_channel > indi[i][3]:
                            true_input_channel = indi[i][3]
                        estimated_out_channel = true_input_channel + indi[i][0] * indi[i][1]

                        if estimated_out_channel == true_out_channel_channel:
                            break
                        else:
                            _in_channel = estimated_out_channel   
        indi.pop(mutation_position)
    return indi

def do_alter_mutation(indi, mutation_position):
    """
            ----out_channel of resnet
            ----amount in one resnet
            ----amount in one densenet
            ----pooling type
    """
    set = ModelSettings()
    if len(indi[mutation_position]) == 1:
        indi[mutation_position][0] = 1 - indi[mutation_position][0]
    else:
        if len(indi[mutation_position]) == 3:
            indi[mutation_position][0] = np.random.randint(set.min_amount, set.max_amount1+1)
            indi[mutation_position][2] = set.output_channles[np.random.randint(0, len(set.output_channles))]
            next_in_channel = indi[mutation_position][2]

        if len(indi[mutation_position]) == 4:
            k = indi[mutation_position][0]
            if k == 12:
                amount = np.random.randint(set.min_amount, set.max_amount1+1)
            if k == 20:
                amount = np.random.randint(set.min_amount, set.max_amount1+1)
            if k == 40:
                amount = np.random.randint(set.min_amount, set.max_amount2+1)
            indi[mutation_position][1] = amount
            true_input_channel = indi[mutation_position][2]
            if true_input_channel > indi[mutation_position][3]:
                true_input_channel = indi[mutation_position][3]
            next_in_channel = true_input_channel + k * amount

        if mutation_position != (len(indi)-1):
            for i in range(mutation_position+1, len(indi)):
                if len(indi[i]) == 3:
                    indi[i][1] = next_in_channel
                    break
                if len(indi[i]) == 4:
                    true_input_channel = indi[i][2]
                    if true_input_channel > indi[i][3]:
                        true_input_channel = indi[i][3]
                    true_out_channel_channel = true_input_channel + indi[i][0] * indi[i][1]
                    
                    indi[i][2] = next_in_channel
                    true_input_channel = indi[i][2]
                    if true_input_channel > indi[i][3]:
                        true_input_channel = indi[i][3]
                    estimated_out_channel = true_input_channel + indi[i][0] * indi[i][1]

                    if estimated_out_channel == true_out_channel_channel:
                        break
                    else:
                        next_in_channel = estimated_out_channel   
    return indi

def delete(NPOP):
    ia = []
    for i in NPOP:
        ia.append(NPOP.index(i))
    ia = list(set(ia))
    POP = copy.deepcopy(select(NPOP, ia))
    return POP

def computing_fitness(it,NPOP, train_load, val_load, epoch_num, LR, trainset, valset, device, num_classes, lossfile, best_loss1, early_stop1, max_early_stop1):
    # ======================================================================================================================
    # device detection
    # ======================================================================================================================
    device = device
    print('device: ', device)
    
    f = np.zeros((len(NPOP), 2))
    obj_f = np.zeros((len(NPOP), 1))
    AUC_f = np.zeros((len(NPOP), 1))
    fin_modelt = []
    for t in range(0, len(NPOP)):
        # ======================================================================================================================
        # model building
        # ======================================================================================================================
        print(NPOP[t])
        model = EvoCNNModel(num_classes, NPOP[t])
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss() 
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=5*1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1) # lr = lr *gamma
        m = nn.Softmax(dim=1)

        train_labels = Variable(torch.zeros(len(trainset), 1).to(device))
        train_preds = Variable(torch.zeros(len(trainset), 1).to(device))
        train_probs = Variable(torch.zeros(len(trainset), 1).to(device))
        train_losses = Variable(torch.zeros(epoch_num, 1).to(device))
        train_metrics = []

        val_labels = Variable(torch.zeros(len(valset), 1).to(device))
        val_preds = Variable(torch.zeros(len(valset), 1).to(device))
        val_probs = Variable(torch.zeros(len(valset), 1).to(device))
        val_losses = Variable(torch.zeros(epoch_num, 1).to(device))
        val_metrics = []

        model_all = []

        # ======================================================================================================================
        # model training, validation, testing
        # ======================================================================================================================
        best_loss = best_loss1
        early_stop = early_stop1
        max_early_stop = max_early_stop1
        
        print("\033[1;32m    it_num [{}] \033[0m".format(it))
        for i in range(1, epoch_num + 1):
            print('\n')
            print("\033[1;32m    Epoch [{}/{}]     itnum [{}]\033[0m".format(i, epoch_num, it))
            print("-" * 45)

            # model training
            model.train()
            train_loss = 0.0
            cprint('r', "model is training...")
            index = 0
            for X, y, in train_load:
                X, y = Variable(X.to(device)), Variable(y.to(device)) # to device
                pred = model(X)                     # prediction, logits
                probs = m(pred)                     # probability
                _, pred_y = torch.max(probs, 1)     # predicted label
                optimizer.zero_grad()
                loss = loss_fn(pred, y)
                loss.backward()                     # backward pass
                optimizer.step()                    # weights update
                train_loss += len(y) * loss.item()  

                train_labels[index:index+len(y), 0] = y
                train_preds[index:index+len(y), 0] = pred_y
                train_probs[index:index+len(y), 0] = probs[:,1]

                index += len(y)

            scheduler.step()  

            print('train loss :', round(1.0*train_loss/len(trainset), 4))
            train_metric = compute_measures(train_labels, train_preds, train_probs)
            train_losses[i - 1, 0] = round(1.0 * train_loss / len(trainset), 4)
            train_metrics.append(train_metric)
            metrics_print(train_metric)
            
            # model validating
            model.eval() #通知dropout层和BN层在train和validation/test模式间
            val_loss = 0.0
            cprint('r', "\nmodel is validating...")
            index = 0
            with torch.no_grad():
                for X, y, in val_load:
                    X, y = Variable(X.to(device)), Variable(y.to(device))
                    pred = model(X)
                    probs = m(pred)
                    _, pred_y = torch.max(probs, 1)
                    loss = loss_fn(pred, y)
                    val_loss += len(y) * loss.item()
                    #import pdb
                    #pdb.set_trace()
                    val_labels[index:index + len(y), 0] = y
                    val_preds[index:index + len(y), 0] = pred_y
                    val_probs[index:index + len(y), 0] = probs[:, 1]

                    index += len(y)
            print('val loss :', round(1.0 * val_loss / len(valset), 4))
            val_metric = compute_measures(val_labels, val_preds, val_probs)
            val_losses[i - 1, 0] = round(1.0 * val_loss / len(valset), 4)
            metrics_print(val_metric)

            if val_losses[i - 1, 0] < best_loss:
                best_loss = val_losses[i - 1, 0]
                early_stop = 0
                # model saving
                tt = copy.deepcopy(model) 
                model_all.append(tt)
                val_metrics.append(val_metric)
            else:
                early_stop += 1
            if early_stop == max_early_stop:
                print("Early Stopping")
                break
        best_model = model_all[-1]

        plot_per_epoch(lossfile, val_losses.cpu().numpy(), 'Valid loss.png', train_losses.cpu().numpy())
        
        f[t][0] = val_metrics[-1]['sen']
        f[t][1] = val_metrics[-1]['spe']
        obj_f[t] = val_metrics[-1]['acc']
        AUC_f[t] = val_metrics[-1]['auc']

        fin_modelt.append(best_model)
    fin_modelt = np.array(fin_modelt)
    return f, obj_f, AUC_f, fin_modelt

def FNDSF(pa):
    N = len(pa)
    C = len(pa[0])
    FL = []
    F = np.zeros((N, N))
    Fpa = np.zeros((N, C + 2))
    Fpa[:, 0: 2] = pa
    S = np.zeros((N, N))
    for p in range(0, N):
        P = pa[p, :]
        Sp = []
        SS = []
        for q in range(0, N):
            Q = pa[q, :]
            if DONtwo(P, Q) == 1:
                Sp.append(q)
            elif DONtwo(Q, P) == 1:
                Fpa[p, C] = Fpa[p, C] + 1
            mm = np.array(Sp)
            S[p, 0:len(Sp)] = np.array(Sp)
            SS.append(Sp)
        if Fpa[p, C] == 0:
            Fpa[p, C + 1] = 1
    i = 1
    isign = [k for k, x in enumerate(Fpa[:, C + 1]) if x == i]
    while len(isign) != 0:
        for p in range(0, len(isign)):
            m=isign[p]
            zzz = S[isign[p], :]
            aaa = [k for k, x in enumerate(zzz) if x == 0]
            for q in range(0, len(zzz)):
                if zzz[q] == 0 and q != 0:
                    break
                else:
                    m=int(zzz[q])
                    Fpa[m, C] = Fpa[m, C] - 1
                    if Fpa[m, C] == 0 or Fpa[m, C] == -1:
                        Fpa[m, C + 1] = i + 1
        i = i + 1
        isign = []
        isign = [k for k, x in enumerate(Fpa[:, C + 1]) if x == i]
    FL = Fpa[:, C + 1]-1
    return FL

def UPPNSGAIIf(POP, pa, FL, NM, f, AUC, model):
    pal = np.c_[pa, FL]
    Epal = []
    EPOP = []
    E_f = []
    E_AUC = []
    E_model = []
    i = 0

    while len(Epal) < NM:

        Fisign = [k for k, x in enumerate(FL) if x == i]
        if len(Epal) + len(Fisign) <= NM:
            Epal.extend(pal[Fisign])
            EPOP.extend(copy.deepcopy(select(POP,Fisign)))
            E_f.extend(f[Fisign])
            E_AUC.extend(AUC[Fisign])
            E_model.extend(model[Fisign])
            i = i + 1
        else:
            Nneed = NM - len(Epal)
            tpal = pal[Fisign]
            tPOP = copy.deepcopy(select(POP,Fisign))
            tf = f[Fisign]
            tAUC = AUC[Fisign]
            tmodel = model[Fisign]

            y = tAUC[:, 0].argsort()
            y = y[len(y)-Nneed:len(y)]
            Epal.extend(tpal[y])
            EPOP.extend(copy.deepcopy(select(tPOP, y)))
            E_f.extend(tf[y])
            E_AUC.extend(tAUC[y])
            E_model.extend(tmodel[y])
            i = i + 1
    Epal = np.array(Epal)
    return EPOP, Epal, E_f, E_AUC, E_model

def select_models(EPOP, Epal, E_f, E_AUC, E_model, min_model):
    E_f = np.array(E_f)
    E_AUC = np.array(E_AUC)
    E_model = np.array(E_model)

    idx = [k for k, x in enumerate(Epal[:, 2]) if x == 0]
    pareto = Epal[idx, 0:2]
    FL = Epal[idx, 2]
    fin_POP = copy.deepcopy(select(EPOP, idx))
    fin_f = E_f[idx]
    fin_AUC = E_AUC[idx]
    fin_model = E_model[idx]
    tmp_label = np.ones((len(pareto), 1))
    for t in range(0, len(pareto)):
        if pareto[t, 0] == 0 or pareto[t, 1] == 0:
            tmp_label[t] = 0
        if pareto[t, 0] / pareto[t, 1] < 0.5 or pareto[t, 0] / pareto[t, 1] > 1.5:
            tmp_label[t] = 0
    idx1 = [k for k, x in enumerate(tmp_label) if x == 1]
    pareto = pareto[idx1]
    FL = FL[idx1]
    fin_POP = copy.deepcopy(select(fin_POP, idx))
    fin_f = fin_f[idx1]
    fin_AUC = fin_AUC[idx1]
    fin_model = fin_model[idx1]
    m = 0

    while len(pareto) < min_model:
        m = m + 1
        idxm = [k for k, x in enumerate(Epal[:, 2]) if x == m]
        pareto = np.r_[pareto, Epal[idxm, 0:2]] 
        FL = np.r_[FL, Epal[idxm, 2]]
        fin_POP.extend(copy.deepcopy(select(EPOP, idxm)))
        fin_f = np.r_[fin_f, E_f[idxm]]
        fin_AUC = np.r_[fin_AUC, E_AUC[idxm]]
        fin_model = np.r_[fin_model, E_model[idxm]]
        tmp_label = np.ones((len(pareto), 1))
        for t in range(0, len(pareto)):
            if pareto[t, 0] == 0 or pareto[t, 1] == 0:
                tmp_label[t] = 0
            elif pareto[t, 0] / pareto[t, 1] < 0.5 or pareto[t, 0] / pareto[t, 1] > 1.5:
                tmp_label[t] = 0
        idx2 = [k for k, x in enumerate(tmp_label) if x == 1]
        pareto = pareto[idx2]
        FL = FL[idx2]
        fin_POP = copy.deepcopy(select(fin_POP, idx2))
        fin_f = fin_f[idx2]
        fin_AUC = fin_AUC[idx2]
        fin_model = fin_model[idx2]
    return pareto, FL, fin_POP, fin_f, fin_AUC, fin_model

def cal_weight(pareto, AUC, lamda):
    w = np.zeros((len(pareto), 1))
    for j in range(0, len(pareto)):
        if pareto[j, 0] >= pareto[j, 1]:
            w[j] = pareto[j, 1] / pareto[j, 0] #修改权重
        else:
            w[j] = pareto[j, 0] / pareto[j, 1]
        w[j] = lamda * w[j] + (AUC[j] / max(AUC)) * [1 - lamda]
    t = sum(w)
    for i in range(0, len(pareto)):
        w[i] = w[i] / t
        w[i] = round(w[i][0],4)
    return w

def ERE(fin_model, w, valset, device, num_classes, T):
    val_load = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
    val_preds = torch.zeros(len(valset), 1)
    val_probs = torch.zeros(len(valset), 1)
    val_uncertainty = torch.zeros(len(valset), 1)

    p, u, val_labels = cal_pu(valset, val_load, fin_model, device, num_classes, T)

    index = 0
    for i in range(len(p[0])):
        sum = 1 + u[:,i]
        ppp = np.zeros((len(fin_model), 2))
        ppp[:,0] = (1-p[:,i])/sum
        ppp[:,1] = p[:,i]/sum
        probs,uncertainty = Analytic_ER_rule(w, ppp)
        probs = list(chain.from_iterable(probs))
        probs = torch.tensor(probs)
        _, pred_y = torch.max(probs, 0)
        val_preds[index:index + 1, 0] = pred_y
        val_probs[index:index + 1, 0] = probs[1]
        val_uncertainty[index:index + 1, 0] = uncertainty[0]

        index += 1
    return val_labels, val_preds, val_probs, val_uncertainty

def cal_pu(valset, val_load, fin_model, device, num_classes, T):
    device = device
    p = np.zeros((len(fin_model), len(valset)))
    u = np.zeros((len(fin_model), len(valset)))
    m = nn.Softmax(dim=1)
    for i in range(0, len(fin_model)):
        model = fin_model[i]
        model.layer.add_module('dropouts', nn.Dropout(0.5))
        model = model.to(device)
        #model[i].eval()
        index = 0
        test_preds = torch.zeros(len(valset), 1)
        test_probs = torch.zeros(len(valset), 1)
        test_labels = torch.zeros(len(valset), 1)
        test_uncertainty = torch.zeros(len(valset), 1)
        with torch.no_grad():
            for X, y, in val_load:
                X, y = Variable(X.to(device)), Variable(y.to(device))
                probs = torch.zeros(len(y), num_classes).to(device)
                for _ in range(T):
                    pred = model(X)
                    probs += m(pred)
                probs = (probs/T).t()
                uncertainty = -probs[0] * math.log(probs[0]) - (1 - probs[0]) * math.log(1 - probs[0])
                _, pred_y = torch.max(probs, 0)
                test_labels[index:index + len(y), 0] = y
                test_probs[index:index + len(y), 0] = probs[1, :]
                test_preds[index:index + len(y), 0] = pred_y
                test_uncertainty[index:index + len(y), 0] = uncertainty
                index += len(y)
        p[i, :] = test_probs.t()
        u[i, :] = test_uncertainty.t()
    return p, u, test_labels

def Analytic_ER_rule(w,p):
    m = len(p)
    n = len(p[0])
    
    sum_p = np.zeros((m,1))
    for j in range(0, m):
        sum_p[j] = p[j][0]
        for i in range(1, n):
            sum_p[j] = sum_p[j] + p[j][i]

    sec1 = np.zeros((1,n))
    for i in range(0, n):
        sec1[0][i] = 1
        for j in range(0, m):
            sec1[0][i] = (w[j] * p[j, i] + 1 - w[j] * sum_p[j]) * sec1[0][i]
    sec2 = 1
    sec3 = 1
    for j in range(0, m):
        sec2 = (1 - w[j]) * sec2
        sec3 = (1 - w[j] * sum_p[j]) * sec3
    
    sum_sec1 = 0
    for i in range(0, n):
        sum_sec1 = sum_sec1 + sec1[0][i]

    k = sum_sec1 - (n - 1) * sec3
    k = 1 / k

    p_fin = []
    for i in range(0, n):
        numerator = k * (sec1[0][i] - sec3)
        denominator = 1 - (k * sec2)
        a = numerator / denominator
        if denominator==0:
            p_fin.append(0)
        else:
            p_fin.append(a)
    if denominator==0:
        u_fin = 1
    else:
        u_fin = k * (sec3 - sec2) / denominator
    return p_fin,u_fin

def cal_single_probability(single_test_data, model, device):
    device = device
    p = np.zeros((len(model), 2))
    m = nn.Softmax(dim=1)
    for i in range(0, len(model)):
        model[i] = model[i].to(device)
        model[i].eval()
        pred = model[i](single_test_data)
        
        probs = m(pred) #tensor device:cuda
        probs = probs.cpu().detach() #tensor device:cpu
        pro = np.array(probs)
        pro_output = np.zeros((len(pro), 2))
        pro_output[:, 0] = pro[:, 0]
        pro_output[:, 1] = pro[:, 1]
        p[i,:]=pro_output
    return p

def CDAf1(tPOP, tpa): # calculate crowding-distancein
    m = len(tpa)
    n = len(tpa[0])
    tp = np.zeros((m, 1))
    for i in range(0, n):
        y = tpa[:, i].argsort()
        tPOP = copy.deepcopy(select(tPOP, y))
        tpa = tpa[y, :]
        tp = tp[y]
        tp[0, 0] = float("inf")
        tp[m - 1, 0] = float("inf")
        if m == 1 or m == 2:
            tpai1 = tpa[:, i]
            tpad1 = tpa[:, i]
        else:
            tpai1 = tpa[2:m, i]
            tpad1 = tpa[0:(m - 2), i]
        tpai1 = tpai1.flatten()
        tpad1 = tpad1.flatten()
        a = abs(tpai1 - tpad1)
        fimin = min(tpa[:, i])
        fimax = max(tpa[:, i])
        if a.any() == 0:
            tp[1:(m - 1), 0] = tp[1:(m - 1), 0]
        else:
            tp[1:(m - 1), 0] = tp[1:(m - 1), 0] + a / (0.0001 + fimax - fimin)
    return tPOP, tpa, tp

def DONtwo(pa1, pa2):
    C = len(pa1)
    Dtrue = 0
    m = pa1[0]
    n = pa1[1]
    pade = pa2 - pa1
    aa = np.argwhere(pade <= 0)
    if len(aa) == C:
        bb = np.argwhere(pade < 0)
        if len(bb) > 0:
            Dtrue = 1
    return Dtrue

def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()

def metrics_print(metrics_dict):
    for key, value in metrics_dict.items():
        print("{:s}: {:.4f}".format(str.upper(key), value), end=' ')

def compute_measures(labels, preds, predict_prob):
    '''
    :param labels: ground truth
    :param preds: predicted label
    :param predict_prob: postive probability (1)
    :return: ACC, SEN, SPE, AUC
    '''
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    predict_prob = predict_prob.cpu().detach()
    cm = confusion_matrix(labels, preds)
    TP = float(cm[1][1])
    TN = float(cm[0][0])
    FP = float(cm[0][1])
    FN = float(cm[1][0])

    sensitivity = TP / ((TP + FN) + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    fpr, tpr, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(fpr, tpr)
    index = [round(sensitivity,4), round(specificity,4), round(roc_auc,4), round(accuracy,4)]
    res_index = dict(zip(['sen', 'spe', 'auc', 'acc'], index))
    return res_index

class ToTensor(object):
    def __call__(self, image):
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # image = image / 255
        image = torch.from_numpy(image)
        return image
class To01(object):
    def __call__(self, image):
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image = image / 255
        return image
class myDataset:
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform  = transform
        self.class_pathes = os.listdir(self.root_path)
        self.data = []
        self.label = []
        for (i,path) in enumerate(self.class_pathes):
            cur_path = os.path.join(self.root_path, path)
            files = os.listdir(cur_path)
            self.data += [os.path.join(cur_path, file) for file in files]
            self.label += [i] * len(files)
        self.label = torch.tensor(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.label[idx]
        img = np.load(img_path)
        if self.transform:
            img = self.transform(img)
            img = img.float() 
        return img, label
class squeeze(object):
    def __call__(self, image):
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image = np.squeeze(image)
        return image
class newaxis(object):
    def __call__(self, image):
        image = image[np.newaxis,:,:,:]
        return image

def plot_per_epoch(ckpt_dir, measurements1, fname, measurements2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements1) + 1), measurements1)
    ax.plot(range(1, len(measurements2) + 1), measurements2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss')
    ax.set_title('loss')
    plt.tight_layout()
    
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()   

def plot_per_it(ckpt_dir, save_pa, fname):
    color = ['r','g','b','k','y','c','m','pink','peru','gray','darkorange']
    for i in range(len(save_pa)):
        x = save_pa[i][:,0]
        y = save_pa[i][:,1]
        t = x.argsort()
        x = x[t]
        y = y[t]
        plt.plot(x,y,c=color[i],label=str(i))
        plt.scatter(x,y,c=color[i])
    plt.legend(loc='best')
    plt.title("optimization")
    plt.xlabel("SEN")
    plt.ylabel("SPE")
    
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()      

def select(EPOP, idx):
    a = []
    for i in range(len(idx)):
        a.append(EPOP[idx[i]])
    return a