# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:39:06 2024
node2vec embeddings before machine learning on condensed DHFR dataset
1. embeddings optimization: graph embedding are obtained by stacking
node mean and node variance.
2. MLP one hidden layer training
@author: hari
"""

import torch
from torch.utils.data import Dataset

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import ToDevice
from torch_geometric.nn import Node2Vec
# nn model
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



# %% load dataset (to device)
composition = ToDevice(ARGS['device'])
dataset = InMemoryDataset(transform= composition)
dataset.load(osp.join(ARGS['root_folder'], ARGS['data_name'],
                      'processed', 'data.pt'))
# infer node labels from graph labels, required by embedding optimization
nodeLabels= torch.cat([g.y*torch.ones([g.num_nodes], device=ARGS['device']) 
                       for g in dataset], dim=0) 
assert nodeLabels.shape[0] == dataset.x.shape[0]


# %% definition of embedding optimization

def embedTrain(model:Node2Vec, loader, optimizer)-> float:
    """
    expects model and loader to be already on the desired device
    """
    model.train()
    total_loss = 0
    #pos_rw, neg_rw = next(iter(loader))# DEBUG
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(ARGS['device']), neg_rw.to(ARGS['device']))
        # BUG in node2vec.py pos/neg_sample return cat dim=1 not 0! 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def embedTest(model:Node2Vec, nodeLabels:Data, idxTrain, idxTest)->float:
    """
    nodeLabels must be inferred from graph label for all nodes
    uses sklearn LogisticRegression under the hood
    """
    model.eval()
    zz = model() # chiamata a funzione rompe i coglioni
    acc = model.test(
        train_z= zz[idxTrain],
        train_y= nodeLabels[idxTrain],
        test_z= zz[idxTest],
        test_y= nodeLabels[idxTest],
        max_iter=ARGS['embed']['test_iter'],
    )
    return acc


def embeddingLoop(dataset:Data, idxTrain, idxTest)-> torch.tensor:
    # define embedding model
    embedModel = Node2Vec(dataset.edge_index, **ARGS['node2vec']).to(ARGS['device'])
    # TODO consider increasing num_workers in model.loader
    embedLoader= embedModel.loader(batch_size= ARGS['embed']['batch_size'],
                                   shuffle= ARGS['embed']['shuffle'],
                                   num_workers= ARGS['embed']['num_workers'])
    embedOptimizer= torch.optim.SparseAdam(list(embedModel.parameters()),
                                           lr=ARGS['embed']['learning_rate'])
    # nodeLabels are computed when loading the dataset
    
    epochs= tqdm(range(ARGS['embed']['max_epochs']), ncols=100)
    for e in epochs:
        loss= embedTrain(embedModel, embedLoader, embedOptimizer)
        acc= embedTest(embedModel, nodeLabels, idxTrain, idxTest)
        desc = f"EMB {e: 4d}| L:{loss:4.3f}| A:{acc:4.3f}"
        epochs.set_description(desc)
    
    
    # stack mean and var of each graph to get graph fatures
    collectionSlices = [g.num_nodes for g in dataset]
    zz = embedModel()
    graphEmbeddings = zz.split(collectionSlices, dim=0)
    
    graphFeatures= torch.stack([torch.cat([ge.mean(dim=0),ge.var(dim=0)])
                        for ge in graphEmbeddings])
    
    return graphFeatures.clone().detach()


# %% define torch model, train and test procedures

class MLP(torch.nn.Module):
    # TODO possible improvement using nn.Sequential
    def __init__(self):
        super().__init__()
        hDim = ARGS['lay2']['hidden_dim']
        # layer 1
        self.lay1 = Linear(in_features= ARGS['node2vec']['embedding_dim']*2,
                              out_features= hDim if hDim else dataset.num_classes)
        if ARGS['lay1']['dropout']:
            self.drop1= F.dropout
        if ARGS['lay1']['activation']:
            self.activ1= ReLU()
        # layer 2
        if hDim:
            self.lay2 = Linear(in_features= hDim,
                               out_features= dataset.num_classes)
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

class ReaderDataset(Dataset):
    """https://pytorch.org/docs/stable/data.html \n
    creates a dataset from computed tensor of embeddings
    """
    def __init__(self, embeddings:torch.Tensor, labels:torch.Tensor ):
        assert embeddings.shape[0] == labels.shape[0]
        self.x = embeddings.to(ARGS['device'])
        self.y = labels.reshape([-1]).to(ARGS['device'])
        # ensure same number of elements
        assert self.x.shape[0] == self.y.shape[0]
        # utility
        self.num_features= self.x.shape[1]
        self.num_classes= len(self.y.unique())
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


def learningLoop(randomSeed:int)-> torch.tensor:
    """
    performs training and return predicted values for best model
    """
    # set random seed
    randGen = torch.manual_seed(randomSeed)
    perm = torch.randperm(len(dataset), generator=randGen)
    b = int(len(dataset)*(1-ARGS['test_share'])) 
    maskTrain, maskTest = perm[:b], perm[b:]
    # compute embeddings: dataset is a GLOBAL variable
    zz = embeddingLoop(dataset, maskTrain, maskTest)
    # clone and detach embeddings tensor to avoid autograd backtrace
    data = ReaderDataset(zz, dataset.y)
    # define NN model
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
    
    epochs= tqdm(range(ARGS['max_epochs']), ncols=100)
    accWindow = torch.zeros([ARGS['ma_window']])
    for e in epochs:
        # FIXME  problem on second run of the loop
        loss = trainLoss(model, optimizer, criterion, data[maskTrain])
        accT = testAccuracy(model, data[maskTrain])
        accV = testAccuracy(model, data[maskTest])
        desc = f"{e: 4d}| L:{loss:4.3f}| T:{accT:4.3f}| V:{accV:4.3f}"
        epochs.set_description(desc)
        # early stopping with moving average update threshold
        accWindow = accWindow.roll(-1)
        accWindow[-1]=accV
        if (accWindow.max()-accWindow.min()) < ARGS['ma_threshold']:
            break
        
    
    # REMARK scores are not needed, computed later wirh information score
    return model(data.x).clone().detach()


# %% MAIN loop 
with open(osp.join(ARGS['root_folder'], ARGS['seed_name']+'.txt'),'r') as f:
    seeds = f.readlines()


seeds = [int(s) for s in seeds]

para = Parallel(n_jobs=1, return_as='generator')

outGen = para(delayed(learningLoop)(s) for s in seeds)

outProbs = list(outGen)


# %% export clasification performance (to binary .pt file)
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