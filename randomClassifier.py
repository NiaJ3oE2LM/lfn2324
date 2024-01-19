# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:10:30 2024
Random classifier for DHFR learning comparison
@author: hari
"""
from torch import manual_seed, empty, ones, save
from torch.distributions import Bernoulli
import os.path as osp

# number of graphs in the original DHFR dataset
NUM_GRAPHS= 756


# %% init data structure and set random seeds, generate samples
rootFolder= 'computed'
with open(osp.join(rootFolder,'seedSequence.txt'),'r') as f:
    seeds = f.readlines()

outputs = empty([len(seeds),NUM_GRAPHS])

for i, seed in enumerate([int(s)for s in seeds]):
    manual_seed(seed)
    # toss coins
    p = ones([1,NUM_GRAPHS])/2
    preds = Bernoulli(p).sample()
    # store result
    outputs[i,:]= preds


# %% export data to binary file

save(outputs,osp.join(rootFolder,'randomClassifier.pt'))
