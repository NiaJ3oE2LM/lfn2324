#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:00:12 2024
assess statistical significance of computed, analytical, node features
Load data from random graph collections stored on disk: compute fatures.
Compute z-scores TODO
Compute p-values TODO
@author: hari
"""

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose
from torch_geometric.datasets import TUDataset

# feture computation
from torch import tensor
from utility import (selectNodeFeats, nodeDegree, closenessCentrality, 
                     betweennessCentrality, nodeClustering)

# statistics
from torch import stack, where, zeros

# job handling
from os import listdir
import os.path as osp
from joblib import Parallel, delayed 
from tqdm import tqdm

# post processing (graphics)
from torch import cat
import  matplotlib.pyplot as plt
import datetime as dt

# configuration and logging
import tomllib


# %% load parameters from toml file
DEVICE='cpu'

with open('statisticalSignificance.toml','rb') as f:
    ARGS = tomllib.load(f)


# %% load all available random collection together (list)
rootFolder=ARGS['root_folder']
allCollections=listdir(rootFolder)
# TODO argument for share of random collections
availableCollections = allCollections[:int(len(allCollections)*ARGS['consider_share'])]
print(f"using first {len(availableCollections)} collections of {len(allCollections)}")
# init lit of InMemoryDataset with (all the same) transforms
composition = Compose([
    selectNodeFeats(tensor([ 0,])), # FIXME implment one-hot decoder
    nodeDegree(),closenessCentrality(),betweennessCentrality(),
    nodeClustering(),
    selectNodeFeats(tensor([1, 2, 3, 4])), # removes init graph label
    ])
randCollections = [InMemoryDataset(transform=composition) for _ in availableCollections]
# load and process the dataset
for i,name in enumerate(availableCollections):
    dataPath = osp.join(rootFolder, name, 'processed','data.pt')
    randCollections[i].load(dataPath)

# define labels for later box plot
featLabels = ARGS['feats_name']

# %% load true dataset and apply same transforms
dataset = TUDataset(root='tmp', name='DHFR', transform= composition)


# %% function definition

def computeZScores(idx:int, trueGraph:Data)-> Data:
    """
    computes z-scores for trueGraph in position idx of original dataset
    uses random collections loaded at global level on the specified folder    
    """
    assert idx < len(randCollections[0])
    tmp = [rc[idx].x for rc in randCollections] # rc: random collection
    nodeFeats = stack(tmp) # num_coll*(num_nodes*num_feats)
    # TODO consider also graph level features
    # TODO definition of z-score on lecture notes p. XX
    return (trueGraph.x - nodeFeats.mean(dim=0))/nodeFeats.std(dim=0)


def computePValues(idx:int, trueGraph:Data)-> Data:
    """
    computes p-values for trueGraph in position idx of original dataset
    by counting the occurrencies and estimating the probability
    uses random collections loaded at global level on the specified folder    
    """
    assert idx < len(randCollections[0])
    tmp = [rc[idx].x for rc in randCollections] # rc: random collection
    nodeFeats = stack(tmp) # num_coll*(num_nodes*num_feats)
    # TODO also negative probability
    ans = where(trueGraph.x > nodeFeats)
    count = zeros(nodeFeats.shape)
    count[ans]= 1    
    # FIXME definition of z-score on lecture notes p. XX
    return count.sum(dim=0) / nodeFeats.shape[0]


# %% run jobs in parallel and store result
# TODO argparse
if __name__ == '__main__':
    para = Parallel(n_jobs= ARGS['num_jobs'],
                    batch_size= ARGS['batch_size'],
                    return_as='generator', )
    zScoreGen = para(delayed(computeZScores)(i,g) for i, g in enumerate(dataset))
    zScores = list(tqdm(zScoreGen, total=len(dataset), desc='z-scores'))
    pValueGen = para(delayed(computePValues)(i,g) for i, g in enumerate(dataset))
    pValues = list(tqdm(pValueGen, total=len(dataset), desc='p-values'))
    

# %% plot results: decide which features are significant
catZScores= cat(zScores, dim=0)
catPValues= cat(pValues, dim=0)
# FIXME NaN values
outFolder='img'
# TODO axes subplot with title and slanted labels
fig, axs= plt.subplots(1,2)
fig.suptitle(f"{ARGS['title']} ({len(availableCollections)} samples)")

# generate violin plot
axs[0].set_title("z-scores")
axs[0].violinplot(catZScores.numpy(), showmeans= True)
axs[1].set_title("p-values (neg)")
axs[1].violinplot(catPValues.numpy(), showmeans= True)

# format axes
for ax in axs:
    ax.set_xticks([i+1 for i in range(len(featLabels))],
                  labels= featLabels, rotation=60)
    # TODO add file names under the label with smaller size
    #ax.set_ylim(0,1)

fig.tight_layout()
# save figure and log
outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
outPath= osp.join(ARGS['out_folder'],f"{ARGS['out_name']}_{outVersion}")
plt.savefig(outPath+'.png')
