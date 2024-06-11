#!/usr/bin/python
# -*- coding: UTF-8 -*-

from Bayobjective import objective
from hyperopt import hp, fmin, tpe, Trials
import config

conf = config.config()
min_lamda = conf.min_lamda
max_lamda = conf.max_lamda
min_mut_rate = conf.min_mut_rate
max_mut_rate = conf.max_mut_rate
min_num_model = conf.min_num_model
max_num_model = conf.max_num_model
step_num_model = conf.step_num_model
max_evals = conf.max_evals
save_path = conf.save_path

space = {
    'lamda': hp.uniform('lamda', min_lamda, max_lamda),
    'mut_rate': hp.uniform('mut_rate', min_mut_rate, max_mut_rate), 
    'min_model': hp.quniform('min_model', min_num_model, max_num_model, step_num_model),  #[)
}

tpe_algo = tpe.suggest

tpe_trials = Trials()
tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials,
                max_evals=max_evals)

print(tpe_best)
print('Minimum loss attained with TPE:    {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
print('\nNumber of trials needed to attain minimum with TPE(lamda):    {}'.format(tpe_trials.best_trial['misc']['idxs']['lamda'][0]))
print('\nNumber of trials needed to attain minimum with TPE(mut_rate):    {}'.format(tpe_trials.best_trial['misc']['idxs']['mut_rate'][0]))
print('\nNumber of trials needed to attain minimum with TPE(min_model):    {}'.format(tpe_trials.best_trial['misc']['idxs']['min_model'][0]))

print('\nBest value of lamda from TPE:    {:.4f}'.format(tpe_best['lamda']))
print('\nBest value of mut_rate from TPE:    {:.4f}'.format(tpe_best['mut_rate']))
print('\nBest value of min_model from TPE:    {:.4f}'.format(tpe_best['min_model']))

loss ='Minimum loss attained with TPE:' + str(tpe_trials.best_trial['result']['loss']) + '\n'
trials1='Number of trials needed to attain minimum with TPE(lamda):' + str(tpe_trials.best_trial['misc']['idxs']['lamda'][0]) + '\n'
trials2='Number of trials needed to attain minimum with TPE(mut_rate):' + str(tpe_trials.best_trial['misc']['idxs']['mut_rate'][0]) + '\n'
trials3='Number of trials needed to attain minimum with TPE(min_model):' + str(tpe_trials.best_trial['misc']['idxs']['min_model'][0]) + '\n'

slamda='Best value of lamda from TPE:' + str(tpe_best['lamda']) + '\n'
smut_rate='Best value of mut_rate from TPE:' + str(tpe_best['mut_rate']) + '\n'
smin_model='Best value of min_model from TPE:' + str(tpe_best['min_model']) + '\n'

ftxt = open(save_path, 'a')
ftxt.write(loss)
ftxt.write(trials1)
ftxt.write(trials2)
ftxt.write(trials3)

ftxt.write(slamda)
ftxt.write(smut_rate)
ftxt.write(smin_model)

ftxt.close()
