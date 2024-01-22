#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:19:11 2023
test PG implementation of GraphSAGE convolution, for graph classification
@author: hari
"""
import torch 
# dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader # graph batches
from torch_geometric.transforms import ToDevice
# NN model definition
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
# readout layer for graph class
from torch_geometric.nn import global_mean_pool
# from torch_geometric.nn.aggr import SoftmaxAggregation

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


# %% load dataset (to device)
composition = ToDevice(ARGS['device'])
dataset = InMemoryDataset(transform= composition)
dataset.load(osp.join(ARGS['root_folder'], ARGS['input_name'],
                      'processed', 'data.pt'))


# %% define ML model 

class GCN(torch.nn.Module):
    def __init__(self):
    # TODO tomls all parameters of the 
        super().__init__()
        hDim = ARGS['conv2']['hidden_dim']
        # layer 1
        self.conv1 = SAGEConv(in_channels= dataset.num_node_features,
                              out_channels= hDim if hDim else dataset.num_classes,
                              aggr= ARGS['conv1']['aggr'],
                              )
        if ARGS['conv1']['dropout']:
            self.drop1= F.dropout
        if ARGS['conv1']['activation']:
            self.activ1= F.relu
        # layer 2
        if hDim:
            self.conv2 = SAGEConv(in_channels= hDim,
                                  out_channels= dataset.num_classes,
                                  aggr= ARGS['conv1']['aggr'],
                                  )
            if ARGS['conv2']['dropout']:
                self.drop2= F.dropout
            if ARGS['conv2']['activation']:
                self.activ2= F.relu

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. node embeddings
        x = self.conv1(x, edge_index)
        if hasattr(self, 'drop1'):
            x = self.drop1(x, p=ARGS['conv1']['dropout'], training=self.training)
        if hasattr(self, 'activ1'):
            x = self.activ1(x)
        if hasattr(self, 'conv2'):
            x = self.conv2(x, edge_index)
        if hasattr(self, 'drop2'):
            x = self.drop2(x, p=ARGS['conv2']['dropout'], training=self.training)
        if hasattr(self, 'activ2'):
            x = self.activ2(x)
        # 2. Readout layer (graph classification)
        # TODO is there only this batch function ?
        x = global_mean_pool(x, batch)
        # 3. final classifier
        return F.log_softmax(x, dim=1)

def trainLoss(model, optimizer, criterion, loader):
    """
    loader data needs to be already in the correct device
    """
    model.train()
    # Iterate in batches over the training dataset.
    lossMean = 0
    for data in loader:
        optimizer.zero_grad() # Clear gradients.
        # Perform a single forward pass.
        out = model(data) 
        # Compute the loss.
        loss = criterion(out, data.y)
        lossMean += loss/len(loader)
        # Derive gradients.
        loss.backward()
        # Update parameters based on gradients.
        optimizer.step() 
    
    # FIXME loss from multiple batches ? mean ?
    return lossMean


def testAccuracy(model,loader):
    """
    loader data needs to be already in the correct device
    """
    model.eval()
    count = 0
    # Iterate in batches over the training/test dataset.
    for data in loader:  
        # compute model prediction
        out = model(data)
        # Use the class with highest probability.
        pred = out.argmax(dim=1)
        # Check against ground-truth labels.
        count += int((pred == data.y).sum())
    
    # Compute ratio of correct predictions.
    return count / len(loader.dataset)  


def modelValidate(model, dataset)-> torch.tensor:
    model.eval()
    tmpLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    out= torch.cat([model(data) for data in tmpLoader], dim=0)
    # predixtions are computed upon export
    return out
    

#%% optimization setup and random seed sequences

def learningLoop(randomSeed:int)-> torch.tensor:
    # define object
    model = GCN().to(ARGS['device'])
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
    # shuffle the hole dataset
    randGen = torch.manual_seed(int(randomSeed))
    perm = torch.randperm(len(dataset), generator=randGen)
    # train- test split
    b = int(len(dataset)*(1-ARGS['test_share'])) 
    dataTrain = dataset[perm][:b]
    dataTest = dataset[perm][b:]
    # graph mini-batch
    trainLoader = DataLoader(dataTrain, batch_size=ARGS['batch_size'],
                             shuffle=False)
    testLoader = DataLoader(dataTest, batch_size=ARGS['batch_size'],
                            shuffle=False)

    epochs= tqdm(range(ARGS['max_epochs']), ncols=100)
    accWindow = torch.zeros([ARGS['ma_window']])
    for e in epochs:
        loss = trainLoss(model, optimizer, criterion, trainLoader)
        accT = testAccuracy(model, trainLoader)
        accV = testAccuracy(model, testLoader)
        desc = f"{e: 4d}| L:{loss:4.3f}| T:{accT:4.3f}| V:{accV:4.3f}"
        epochs.set_description(desc)
        # early stopping with moving average update threshold
        accWindow = accWindow.roll(-1)
        accWindow[-1]=accV
        if (accWindow.max()-accWindow.min())< ARGS['ma_threshold']:
            break
    
    # FIXME dataset global variable
    return modelValidate(model, dataset)

# %% MAIN loop

with open(osp.join(ARGS['root_folder'],ARGS['seed_name']+'.txt'),'r') as f:
    seeds = f.readlines()
    dtype=torch.long


seeds = [int(s) for s in seeds]

para = Parallel(n_jobs=2, return_as='generator')

outGen = para(delayed(learningLoop)(s) for s in seeds)

outProbs = list(tqdm(outGen, ncols=100, total=len(seeds)))


# %% export clasification performance (to binaty .pt file)
if ARGS['save']:
    # the information score requires predicted probabilities!
    result = torch.stack(outProbs)
    outVersion= dt.datetime.now().strftime("%y%m%d-%H%M%S")
    outPath= osp.join(ARGS['root_folder'], ARGS['out_folder'], outVersion)
    torch.save(result, outPath+'.pt')
    # print training information
    with open(outPath+'.log','w') as f :
        pprint(ARGS, stream=f)
        pprint(GCN(), stream=f)
    
    
