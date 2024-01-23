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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
# use only the secified share from all available random collections
availableCollections = allCollections[:int(len(allCollections)*ARGS['consider_share'])]
print(f"using first {len(availableCollections)} collections of {len(allCollections)}")
# init lit of InMemoryDataset with (all the same) transforms
composition = Compose([
    selectNodeFeats(tensor([ 0,])), # one-hot decoder
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

def computeZScores(idx:int, trueGraph:Data)-> tensor:
    """
    computes z-scores for trueGraph in position idx of original dataset
    uses random collections loaded at global level on the specified folder    
    """
    assert idx < len(randCollections[0])
    # num_coll*(num_nodes*num_feats) rc: random collection
    nodeFeats = stack([rc[idx].x for rc in randCollections], dim=0) 
    # TODO consider also graph level features
    # TODO definition of z-score on lecture notes p. XX
    ans = (trueGraph.x - nodeFeats.mean(dim=0))/nodeFeats.std(dim=0)
    # set nan values to 0 (non contributing)
    ans[where(ans.isnan())] = 0
    return ans


def computePValues(idx:int, trueGraph:Data)-> (tensor,tensor):
    """
    computes p-values for trueGraph in position idx of original dataset
    by counting the occurrencies and estimating the probability
    uses random collections loaded at global level on the specified folder    
    """
    assert idx < len(randCollections[0])
    # num_coll*(num_nodes*num_feats) rc: random collection
    nodeFeats = stack([rc[idx].x for rc in randCollections], dim=0) 
    # probability: true > random 
    ans = where(trueGraph.x >= nodeFeats)
    countPos = zeros(nodeFeats.shape)
    countPos[ans]= 1
    # probability: true < random 
    ans = where(trueGraph.x < nodeFeats)
    countNeg = zeros(nodeFeats.shape)
    countNeg[ans]= 1  
    # FIXME definition of p-value on lecture notes p. XX
    prPos= countPos.sum(dim=0) / nodeFeats.shape[0]
    prNeg= countNeg.sum(dim=0) / nodeFeats.shape[0]
    return stack([prPos, prNeg], dim=0)


# %% run jobs in parallel and store result
if __name__ == '__main__':
    para = Parallel(n_jobs= ARGS['num_jobs'],
                    batch_size= ARGS['batch_size'],
                    return_as='generator', )
    zScoreGen = para(delayed(computeZScores)(i,g) for i, g in enumerate(dataset))
    zScores = list(tqdm(zScoreGen, total=len(dataset), desc='z-scores'))
    pValueGen = para(delayed(computePValues)(i,g) for i, g in enumerate(dataset))
    pValues = list(tqdm(pValueGen, total=len(dataset), desc='p-values'))
    

# %% plot results: decide which features are significant
# z-Scores: format positive and negative probs for seaborn
catZScores= cat(zScores, dim=0)
n, d = catZScores.shape
flatZScores= catZScores.flatten()
flatZLabels= (tensor([range(d)])+1).repeat([n,1]).flatten()
assert flatZScores.shape == flatZLabels.shape

# p-Values: format positive and negative probs for seaborn
catPValues= cat(pValues, dim=1)

_, n, d = catPValues.shape
flatPosPValues= catPValues[0].flatten()
flatPosLabels= (tensor([range(d)])+1).repeat([n,1]).flatten()
flatPosHues= zeros(flatPosPValues.shape)
assert flatPosPValues.shape == flatPosLabels.shape
flatNegPValues= catPValues[1].flatten()
flatNegLabels= (1+tensor([range(d)])).repeat([n,1]).flatten()
flatNegHues= zeros(flatNegPValues.shape)+1
assert flatNegPValues.shape == flatNegLabels.shape

# FIXME NaN values, drop them ?
outFolder='img'
# TODO axes subplot with title and slanted labels
fig, axs= plt.subplots(1,2)
fig.suptitle(f"{ARGS['title']} ({len(availableCollections)} samples)")

# https://seaborn.pydata.org/generated/seaborn.violinplot.html
axs[0].set_title("z-scores")
violinSetup = {
    'gridsize': len(availableCollections)//2,
    'density_norm':'count',
    'inner':'quart',
    'legend': False,
    }
sns.violinplot(y= flatZScores.numpy(),
               x= flatZLabels.numpy(),
               ax= axs[0], **violinSetup)
axs[1].set_title("p-values")
violinSetup = {
    'gridsize': len(availableCollections)//2,
    'density_norm':'count',
    'split': True,
    'dodge': True,
    'gap': 0., 
    'inner':'quart',
    'legend': False,
    }
sns.violinplot(y= cat([flatPosPValues, flatNegPValues]).numpy(),
               x= cat([flatPosLabels, flatNegLabels]).numpy(),
               hue= cat([flatPosHues, flatNegHues]).numpy(),
               ax= axs[1],**violinSetup)


# format axes
for ax in axs:
    ax.set_xticks([i for i in range(len(featLabels))],
                  labels= featLabels, rotation=90)
    # TODO add file names under the label with smaller size
    #ax.set_ylim(0,1)

fig.tight_layout()
# %% save figure and log
outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
outPath= osp.join(ARGS['out_folder'],f"{ARGS['out_name']}_{outVersion}")
plt.savefig(outPath+'.png')
