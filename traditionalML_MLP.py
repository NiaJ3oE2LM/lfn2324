# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:39:06 2024
traditional machine learning on condensed DHFR dataset
1. MLP
@author: hari
"""

import torch
from torch.utils.data import Dataset, random_split

from joblib import Parallel, delayed 

import datetime as dt
import os.path as osp
from tqdm import tqdm

# TODO argparse hyperparameters and file locations
DEVICE= 'cuda'
TEST_SHARE = 0.3
assert TEST_SHARE>0 and TEST_SHARE<1
MAX_EPOCHS = 1000
assert type(MAX_EPOCHS)==int and MAX_EPOCHS>0
MA_WINDOW= 50
assert type(MA_WINDOW)==int and MA_WINDOW>0
HIDDEN_DIM= 30
ROOT_FOLDER='computed'

# %% load dataset
# TODO possible improvement usinf TensorDataset 
class ReaderDataset(Dataset):
    """https://pytorch.org/docs/stable/data.html \n
    creates a dataset by reading data x and y data from two binary .pt files
    """
    def __init__(self, rootFolder:str, baseName:str):
        self.x = torch.load(osp.join(rootFolder,baseName+'_input.pt')
                            ).to(DEVICE)
        self.y = torch.load(osp.join(rootFolder,baseName+'_output.pt'),
                            ).reshape([-1]).to(DEVICE)
        # ensure same number of elements
        assert self.x.shape[0] == self.y.shape[0]
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


data = ReaderDataset(ROOT_FOLDER,'20240115-105053')

# %% define torch model, train and test procedures

class MLP(torch.nn.Module):
    # TODO possible improvement using nn.Sequential
    def __init__(self, hdim=int):
        super().__init__()
        # random seed initialized glibal level for shuffling
        self.linear1 = torch.nn.Linear(14, hdim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hdim, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
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

def learningLoop(hdim:int, max_epochs:int, randomSeed:int)-> torch.tensor:
    """
    performs training and return predicted values for best model
    """
    # define objects
    model= MLP(hdim).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # set random seed
    randGen = torch.manual_seed(int(randomSeed))
    # train & test split
    dataTrain , dataTest = random_split(data,
                                        [1-TEST_SHARE, TEST_SHARE],
                                        randGen)

    epochs= tqdm(range(MAX_EPOCHS))
    accWindow = torch.zeros([MA_WINDOW])
    for e in epochs:
        loss = trainLoss(model, optimizer, criterion, data[dataTrain.indices])
        acc = testAccuracy(model, data[dataTest.indices])
        epochs.set_description(f"{e}\t Lo:{loss:.4f}\t Ac:{acc:.4f}")
        # early stopping with moving average update threshold
        accWindow = accWindow.roll(-1)
        accWindow[-1]=acc
        if (accWindow.max()-accWindow.min())<1e-3:
            break
        
    
    # REMARK scores are not needed, computed later wirh information score
    return model(data.x)


# %% MAIN loop 
with open(osp.join(ROOT_FOLDER,'seedSequence.txt'),'r') as f:
    seeds = f.readlines()
    dtype=torch.long


seeds = [int(s) for s in seeds]

para = Parallel(n_jobs=2, return_as='generator')

outGen = para(delayed(learningLoop)(HIDDEN_DIM,MAX_EPOCHS,s) for s in seeds)
#graph = dataset[0]# DEBUG 
outPreds = list(tqdm(outGen, total=len(seeds)))


# %% export clasification performance (to binaty .pt file)
result = torch.stack(outPreds).argmax(dim=2)
outVersion= dt.datetime.now().strftime("%Y%m%d-%H%M%S")
outPath= osp.join(ROOT_FOLDER,'traditionalML_')+outVersion
torch.save(result, outPath+'.pt')
# print training information
with open(outPath+'.log','w') as f :
    print(MLP(HIDDEN_DIM), file=f)
    print(f"max epochs: {MAX_EPOCHS}, moving average: {MA_WINDOW}", file=f)

