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
from torch import load, stack, tensor, where, zeros, inner, log2, any

import datetime as dt
import os.path as osp

# post processing (graphics)
import  matplotlib.pyplot as plt

# configuration and logging
import tomllib
from pprint import pprint
import argparse

DEVICE='cpu'

# %% load parameters from toml file

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
modelProbs = stack([load(modelPath).to(DEVICE) for modelPath in modelPaths])
modelTrues = load(osp.join(ARGS['root_folder'], ARGS['true_data']+'.pt'))


# %% define information score

def naiveScore(predProbs:tensor, trueClasses:tensor) -> tensor:
    """
    simply count errors, condenses the second dimension to 1
    """
    predictions= predProbs.argmax(dim=2)
    assert predictions.shape[1] == trueClasses.shape[1]
    assert len(predictions.shape)==2
    # r: random seeds, n: graphs in DHFR (756)
    r,n = predictions.shape
    count = zeros([r,n])
    count[where(predictions == trueClasses.repeat([r,1]))]=1
    return count.sum(dim=1)/n


def informationScore(predProbs:tensor, trueClasses:tensor) -> tensor:
    """ [KonBra91]
    predProbs (expected )shape: num_seeds * num_graphs * num_classes
    trueClasses (expected) shape: 1* num_graphs
    """
    assert predProbs.shape[1] == trueClasses.shape[1]
    assert predProbs.shape[2]==2 # classes {0,1}
    # deal with negative values of log_softmax
    if any(predProbs<0):
        predProbs= predProbs.exp()
    # ensure probability normalization
    predProbs = predProbs / predProbs.sum(dim=2, keepdim= True).repeat([1,1,2])
    assert predProbs[0,0].sum().round()==1
    # r: random seeds, n: graphs in DHFR (756)
    r, ng, _ = predProbs.shape
    # prior distributions (relative frequencies)
    priorCount = zeros(predProbs.shape)
    for c in range(2):
        _, j = where(trueClasses==c)
        priorCount[:,j,c]= 1
    
    freq = priorCount.count_nonzero(dim=1)/ng
    # compute entropy
    entr = -inner(freq[0], log2(freq[0]))
    prior = stack([freq for i in range(ng)], dim=1)
    # information score eq. (4)(5)(6) -> relative average eq. (7)(8)
    score= zeros(predProbs.shape) # one for each random seed
    for c in range(2): # loop available classes
        # j identifies the graphs among the 756
        _, k= where(trueClasses==c)
        # i identifies the samples among the seed sequence
        post_c = predProbs[:,k,c]
        prior_c = prior[:,k,c]
        i, j = where(post_c >= prior_c)
        if len(i)>0 : # post > prior (useful prediction)
            score[i,j,c] = -log2(prior_c[i,j])+ log2(post_c[i,j])
        i, j = where(post_c < prior_c)
        if len(i)>0 : # post < prior (misleading prediction)
            score[i,j,c] = log2(1-prior_c[i,j])- log2(1-post_c[i,j])
    
    return score.sum(dim=[1,2])/ng/entr


# %% compute score
scoresNaive= stack([naiveScore(probs,modelTrues.t())
                    for probs in modelProbs.detach()])

scoresInfo= stack([informationScore(probs,modelTrues.t())
                   for probs in modelProbs.detach()])

# %% generate box and whiskers plot and save
fig, axs= plt.subplots(1,2)
fig.suptitle(f"{ARGS['title']} ({modelProbs.shape[1]} seeds)")

# generate box plot
axs[0].set_title("correct predictions")
axs[0].boxplot(scoresNaive.t().numpy())
axs[1].set_title("information score")
axs[1].boxplot(scoresInfo.t().numpy())

# format axes
lbs = [l for l in modelDefs.keys()]
for ax in axs:
    ax.set_aspect('auto')
    ax.set_xticks([i+1 for i in range(len(lbs))],labels=lbs, rotation=90)
    # TODO add file names under the label with smaller size
    

fig.tight_layout()
# save figure and log
outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
outPath= osp.join(ARGS['out_folder'],f"{ARGS['out_name']}_{outVersion}")
plt.savefig(outPath+'.png', dpi= ARGS['dpi'])
with open(outPath+'.log','w') as f:
    pprint(modelDefs, stream=f)


