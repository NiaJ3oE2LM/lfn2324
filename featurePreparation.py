# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:02:21 2024
Feature pre-processing before machine learning operations. 
Saves data to binary files for later usage.
1. node features. TODO
2. graph features TODO
@author: hari
"""
from torch import tensor, cat, stack, Size, save

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from utility import (onehotDecode, nodeDegree, closenessCentrality, 
                     betweennessCentrality, )

# store collection of graphs to binary file (InMemoryDataset)
from randomGraph_softSwap import storeDataset
import datetime as dt
import os.path as osp
from torch_geometric.data import InMemoryDataset # debug

# condensation transform
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# %% load dataset with chosen features
composition = Compose([
    onehotDecode(3, tensor(list(range(53)))),
    nodeDegree(), closenessCentrality(), betweennessCentrality(),
    ])
dataset = TUDataset(root='tmp', name='DHFR', use_node_attr=True,
                    transform= composition)
# sanity check: assert num_feats
assert dataset[0].x.shape[1] == 7


# %% export graph collection as torch_geometric file (like random collections)
# create list of Data
prepCollection = [d for d in dataset]
# store result as PyG Dataset
rootFolder='computed'
outName= dt.datetime.now().strftime("%Y%m%d-%H%M%S")
rootPath= osp.join(rootFolder,outName)
storeDataset(rootPath, prepCollection)

# sanity check: load exported data and check features
checkDataset = InMemoryDataset()
checkDataset.load(osp.join(rootPath,'processed','data.pt'))
for i in range(len(dataset)):
    assert checkDataset[i].x.shape == dataset[i].x.shape


# %% define condensation transform

def graphFeatures(nodeFeats : tensor)-> tensor:
    """
    condense node feature matrix to graph feature vector
    1. computes the node feature sample mean
    2. appends the node feature sample variance (std)
    """
    graphFeats = cat((nodeFeats.mean(dim=0), nodeFeats.std(dim=0)), dim=0)
    return graphFeats


# %% condense node information to graph-level features
# compute graph features
inputDataset = stack([graphFeatures(graph.x) for graph in dataset])
# sanity check
assert inputDataset.shape == Size([len(dataset),2*7])
# get graph labels
outputDataset = stack([graph.y for graph in dataset])
# sanity check
assert outputDataset.shape == Size([len(dataset),1])


# %% export dataset to standard torch for traditional machine learning
# FIXME one or two files ? two for the moment
save(inputDataset, rootPath+'_input.pt')
save(outputDataset, rootPath+'_output.pt')
