# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:39:06 2024
traditional machine learning on condensed DHFR dataset
1. MLP
@author: hari
"""

import torch
from torch.utils.data import Dataset, random_split
# model definition 
from torch.nn import Linear, ReLU, Softmax
import torch.nn.functional as F

from joblib import Parallel, delayed 
import datetime as dt
import os.path as osp
from tqdm import tqdm

# configuration and logging
import tomllib
from pprint import pprint
import argparse

# %% load parameters from toml file
parser = argparse.ArgumentParser()
parser.add_argument("configFilePath")
tmp = parser.parse_args()

print(tmp.configFilePath)

with open(tmp.configFilePath,'rb') as f:
    ARGS = tomllib.load(f)


# %% load dataset
# TODO possible improvement usinf TensorDataset 
class ReaderDataset(Dataset):
    """https://pytorch.org/docs/stable/data.html \n
    creates a dataset by reading data x and y data from two binary .pt files
    """
    def __init__(self, rootFolder:str, baseName:str):
        self.x = torch.load(osp.join(rootFolder,baseName+'_input.pt')
                            ).to(ARGS['device'])
        self.y = torch.load(osp.join(rootFolder,baseName+'_output.pt'),
                            ).reshape([-1]).to(ARGS['device'])
        # ensure same number of elements
        assert self.x.shape[0] == self.y.shape[0]
        # utility
        self.num_features= self.x.shape[1]
        self.num_classes= len(self.y.unique())
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


data = ReaderDataset(ARGS['root_folder'],ARGS['data_name'])

# %% define torch model, train and test procedures

class MLP(torch.nn.Module):
    # TODO possible improvement using nn.Sequential
    def __init__(self):
        super().__init__()
        hDim = ARGS['lay2']['hidden_dim']
        # layer 1
        self.lay1 = Linear(in_features= data.num_features,
                              out_features= hDim if hDim else data.num_classes)
        if ARGS['lay1']['dropout']:
            self.drop1= F.dropout
        if ARGS['lay1']['activation']:
            self.activ1= ReLU()
        # layer 2
        if hDim:
            self.lay2 = Linear(in_features= hDim,
                               out_features= data.num_classes)
            if ARGS['lay2']['dropout']:
                self.drop2= F.dropout
            if ARGS['lay2']['activation']:
                self.activ2= ReLU()
        # posterior probabilities
        self.softmax = Softmax(dim=1)
        
    def forward(self, x):
        x = self.lay1(x)
        if hasattr(self, 'drop1'):
            x = self.drop1(x, p=ARGS['lay1']['dropout'], training=self.training)
        if hasattr(self, 'activ1'):
            x = self.activ1(x)
        if hasattr(self, 'lay2'):
            x = self.lay2(x)
        if hasattr(self, 'drop2'):
            x = self.drop2(x, p=ARGS['lay2']['dropout'], training=self.training)
        if hasattr(self, 'activ2'):
            x = self.activ2(x)
        return self.softmax(x)
    

def trainLoss(model, optimizer, criterion, data)-> float:
    r"do one step and return the loss"
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    x, y = data
    # Perform a single forward pass.
    out = model(x)
    # Compute the loss solely based on the training nodes.
    loss = criterion(out, y)
    # Derive gradients.
    loss.backward()
    optimizer.step()  # Update parameters based on gradients.
    return loss


def testAccuracy(model, data):
    model.eval()
    x, y= data
    out = model(x)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    count = torch.zeros(y.shape)
    count[torch.where(pred == y)]= 1
    acc= count.sum() / len(y)
    return acc


# %% optimization setup and define (load) seed sequence

def learningLoop(randomSeed:int)-> torch.tensor:
    """
    performs training and return predicted values for best model
    """
    # define objects
    model= MLP().to(ARGS['device'])
    criterion = torch.nn.CrossEntropyLoss()
    match ARGS['opt']['name'].lower():
        case 'adam':
            # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
            optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ARGS['opt']['learning_rate'],
                                 weight_decay= ARGS['opt']['weight_decay'],
                                 )
        case 'sgd':
            # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
            optimizer = torch.optim.SGD(model.parameters(),
                                 lr= ARGS['opt']['learning_rate'],
                                 nesterov= ARGS['opt']['nesterov'],
                                 momentum= ARGS['opt']['momentum'],
                                 dampening= ARGS['opt']['dampening'],
                                 weight_decay= ARGS['opt']['weight_decay'],
                                 )
    # set random seed
    randGen = torch.manual_seed(int(randomSeed))
    # train & test split
    dataTrain , dataTest = random_split(data,
                                        [1-ARGS['test_share'], ARGS['test_share']],
                                        randGen)

    epochs= tqdm(range(ARGS['max_epochs']), ncols=100)
    accWindow = torch.zeros([ARGS['ma_window']])
    for e in epochs:
        loss = trainLoss(model, optimizer, criterion, data[dataTrain.indices])
        accT = testAccuracy(model, data[dataTrain.indices])
        accV = testAccuracy(model, data[dataTest.indices])
        desc = f"{e: 4d}| L:{loss:4.3f}| T:{accT:4.3f}| V:{accV:4.3f}"
        epochs.set_description(desc)
        # early stopping with moving average update threshold
        accWindow = accWindow.roll(-1)
        accWindow[-1]=accV
        if (accWindow.max()-accWindow.min()) < ARGS['ma_threshold']:
            break
        
    
    # REMARK scores are not needed, computed later wirh information score
    return model(data.x)


# %% MAIN loop 
with open(osp.join(ARGS['root_folder'],ARGS['seed_name']+'.txt'),'r') as f:
    seeds = f.readlines()


seeds = [int(s) for s in seeds]
    
para = Parallel(n_jobs=ARGS['num_jobs'], return_as='generator')

outGen = para(delayed(learningLoop)(s) for s in seeds)

outProbs = list(outGen)


# %% export clasification performance (to binaty .pt file)
if ARGS['save'] :
    # the information score requires predicted probabilities!
    result = torch.stack(outProbs)
    outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
    outPath= osp.join(ARGS['root_folder'],ARGS['out_folder'], outVersion)
    torch.save(result, outPath+'.pt')
    # print training information
    with open(outPath+'.log','w') as f :
        pprint(ARGS, stream=f)
        pprint(MLP(), stream=f)