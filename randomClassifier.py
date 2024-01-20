# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:10:30 2024
Random classifier for DHFR learning comparison
@author: hari
"""
from torch import load, manual_seed, rand, empty, save
from torch import tensor, zeros, where, stack, ones

import datetime as dt
import os.path as osp
import tomllib

# %% load configuration and true classes
with open('randomClassifier.toml','rb') as f:
    ARGS = tomllib.load(f)


modelTrues = load(osp.join(ARGS['root_folder'], ARGS['true_name']+'.pt'))
NUM_GRAPHS = len(modelTrues)

# %% init data structure and set random seeds, generate samples
with open(osp.join(ARGS['root_folder'],ARGS['seed_name']+'.txt'),'r') as f:
    seeds = f.readlines()

outProbs = empty([len(seeds), NUM_GRAPHS, 2])

for i, seed in enumerate([int(s) for s in seeds]):
    randGen = manual_seed(seed)
    # toss coins
    preds = rand([NUM_GRAPHS,2], generator= randGen)
    # store result
    outProbs[i]= preds


# %% export data to binary file
outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
outPath= osp.join(ARGS['root_folder'], ARGS['out_folder'], outVersion)
save(outProbs, outPath+'.pt')


# %% fake predixtions to debug information score
# adapted from classificationPerformance.py
priorCount = zeros([len(seeds),NUM_GRAPHS, 2])
for c in range(2):
    j, _ = where(modelTrues==c)
    priorCount[:,j,c]= 1

freq = priorCount.count_nonzero(dim=1)/NUM_GRAPHS
prior = stack([freq for i in range(NUM_GRAPHS)], dim=1)

usefulProbs = prior.clone().detach()
misleadingProbs = prior.clone().detach()
for c in range(2):
    j, _ = where(modelTrues==c)
    usefulProbs[:,j,c] += ARGS['eps']
    misleadingProbs[:,j,c] -= ARGS['eps']
    
correctProbs = priorCount
incorrectProbs = ones(correctProbs.shape)-correctProbs

# %% export data to binary file
outPath= osp.join(ARGS['root_folder'], ARGS['out_folder'],'')
save(correctProbs, outPath+'correct.pt')
save(incorrectProbs, outPath+'incorrect.pt')
save(usefulProbs, outPath+'useful.pt')
save(misleadingProbs, outPath+'misleading.pt')
