# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:43:42 2024
Compare classification performance of different predictors using the
the information score defined in [KonBra91](https://doi.org/10.1007/BF00153760)
Considered Models
1. random classifier
2. traditional machine learning (MLP)
3. embedding machine learning (MLP)
4. Simple Graph Convolution
5. GraphSAGE
@author: hari
"""
from torch import load, stack, tensor, where, zeros, inner, log2

import datetime as dt
import os.path as osp

# post processing (graphics)
import  matplotlib.pyplot as plt

# configuration and logging
import tomllib
from pprint import pprint
import argparse


# %% load parameters from toml file
DEVICE='cpu'

parser = argparse.ArgumentParser()
parser.add_argument("configFilePath")
tmp = parser.parse_args()

with open(tmp.configFilePath,'rb') as f:
    ARGS = tomllib.load(f)


# %% define and load models
modelDefs = ARGS['model_defs']
modelPaths = [osp.join(ARGS['root_folder'],dataFile+'.pt')
              for _, dataFile in modelDefs.items()]


# %% load original dataset and prediction data for all models
""" idea
define a tensor of dimension 3 where the indeces have the following meaning:
    1. model index
    2. sample among different random seeds
    3. DHFR graph index <- to be condensed by the performance score
"""
modelPreds = stack([load(modelPath).to(DEVICE) for modelPath in modelPaths])
modelTrues = load(osp.join(ARGS['root_folder'], ARGS['true_data']+'.pt'))


# %% define information score

def naiveScore(predictions:tensor, trueClasses:tensor) -> tensor:
    """
    simply count errors, condenses the second dimension to 1
    """
    assert predictions.shape[1] == trueClasses.shape[1]
    assert len(predictions.shape)==2
    # n: random seeds, d: graphs in DHFR (756)
    n,d = predictions.shape
    count = zeros([n,d])
    count[where(predictions == trueClasses.repeat([n,1]))]=1
    return count.sum(dim=1)/d


def informationScore(predictions:tensor, trueClasses:tensor) -> tensor:
    """ [KonBra91]
    """
    assert predictions.shape[1] == trueClasses.shape[1]
    assert len(predictions.shape)==2
    # n: random seeds, d: graphs in DHFR (756)
    n,d = predictions.shape
    # prior distributions (relative frequencies)
    priorCount = zeros([2,1,d])
    i, j = where(trueClasses==0)
    priorCount[0,i,j]= 1
    i, j= where(trueClasses==1)
    priorCount[1,i,j]= 1
    prior = tensor([priorCount[0].count_nonzero(dim=1), 
                    priorCount[1].count_nonzero(dim=1)])/d
    # compute entropy
    entr = -inner(prior, log2(prior))
    prior = prior.repeat([n,1])
    # posterior distributions (relative frequencies)
    postCount = zeros([2,n,d])
    i, j = where(predictions==0)
    postCount[0,i,j]= 1
    i, j= where(predictions==1)
    postCount[1,i,j]= 1
    post = stack([postCount[0].count_nonzero(dim=1), 
                  postCount[1].count_nonzero(dim=1)], dim=1)/d
    # information score eq. (4)(5)(6) -> relative average eq. (7)(8)
    score= zeros([n]) # one for each random seed
    assert prior.shape==post.shape
    for c in range(2): # loop available classes
        # j identifies the graphs among the 756
        _, j= where(trueClasses==c)
        # i identifies the samples among the seed sequence
        i = where(post[:,c] > prior[:,c])
        if len(i[0])>0 : # post > prior (useful prediction)
            score[i] += (-log2(prior[i][:,c])+ log2(post[i][:,c]))*len(j)/d
        i = where(post[:,c]<=prior[:,c])
        if len(i[0])>0 : # post < prior (misleading prediction)
            score[i] += (log2(1-prior[i][:,c])- log2(1-post[i][:,c]))*len(j)/d
    
    return score/entr


# %% compute score
scoresNaive= stack([naiveScore(preds,modelTrues.t()) for preds in modelPreds])

scoresInfo= stack([informationScore(preds,modelTrues.t()) for preds in modelPreds])

# %% generate box and whiskers plot and save
fig, axs= plt.subplots(1,2)
fig.suptitle(f"{ARGS['title']} ({modelPreds.shape[1]} seeds)")

# generate box plot
axs[0].set_title("correct predictions")
axs[0].boxplot(scoresNaive.t().numpy())
axs[1].set_title("information score")
axs[1].boxplot(scoresInfo.t().numpy())

# format axes
lbs = [l for l in modelDefs.keys()]
for ax in axs:
    ax.set_xticks([i+1 for i in range(len(lbs))],labels=lbs, rotation=60)
    # TODO add file names under the label with smaller size
    #ax.set_ylim(0,1)

fig.tight_layout()
# save figure and log
outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
outPath= osp.join(ARGS['out_folder'],f"{ARGS['out_name']}_{outVersion}")
plt.savefig(outPath+'.png')
with open(outPath+'.log','w') as f:
    pprint(modelDefs, stream=f)


