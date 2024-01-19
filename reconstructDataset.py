#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:55:36 2023
reconstruct DHFR datasets because TUD import does not work correctly
@author: hari
"""
# dataset
from torch_geometric.datasets import TUDataset

import torch

#%%  load dataset

subFolder = 'tmp'
dataAlias = 'DHFR'
dataset = TUDataset(root=subFolder, name=dataAlias, use_node_attr=True)

#%% define location of raw files and sanity check (same number of lines)

fileGI = '/'.join([subFolder,dataAlias,'raw',
                    '_'.join([dataAlias,'graph_indicator.txt'])])
fileNL = '/'.join([subFolder,dataAlias,'raw',
                    '_'.join([dataAlias,'node_labels.txt'])])
fileNA = '/'.join([subFolder,dataAlias,'raw',
                    '_'.join([dataAlias,'node_attributes.txt'])])
# no edge files for DHFR dataset !

num_lines = list()

for filePath in [fileGI, fileNL, fileNA]:
    with open(filePath) as f:
        num_lines.append(sum(1 for _ in f))


assert all(num_lines[i] == num_lines[0] for i in range(len(num_lines)))

#%% feature matrix:reconstruct node label one-hot encoding

# initialize container objects
for gi, graph in enumerate(dataset):
    print(gi)
    # init empty label tensor
    dataset[gi]._store['node_labels'] = torch.zeros((graph.x.shape[0],1))
    # init empty position tensor
    dataset[gi].node_positions = torch.zeros((graph.x.shape[0],3))



#%% understand why TU does not work
""" copied from: torch_geometric/io/tu.py
"""
from torch_geometric.io import read_txt_array
import os.path as osp

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

""""""

from torch_geometric.io import read_tu_data
from toch_geometric.utils import one_hot

name ='DHFR'
folder='/'.join(['tmp',name,'raw'])

data, slices, sizes =  read_tu_data(folder, name)

node_labels = read_file(folder, name, 'node_labels', torch.long)
