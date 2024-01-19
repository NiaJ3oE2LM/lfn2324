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

from joblib import Parallel, delayed 

import datetime as dt
import os.path as osp
from tqdm import tqdm

# TODO argparse hyperparameters and file locations
DEVICE= 'cuda'
TEST_SHARE = 0.3
assert TEST_SHARE>0 and TEST_SHARE<1
EMB_EPOCHS = 10 # effective
MLP_EPOCHS = 10 # max, may be broken by moving average filter
embedParams = {
    'embedding_dim':32,
    'walk_length':2,
    'context_size':1,#pos_rw, neg_rw = next(iter(loader))# DEBUG
    'walks_per_node':2,
    'num_negative_samples':2,
    'p':1.0,
    'q':1.0,
    'sparse':True,
    }
MA_WINDOW= 50
MA_THR= 1e-3 # threshold
assert type(MA_WINDOW)==int and MA_WINDOW>0
HIDDEN_DIM= 30
ROOT_FOLDER='computed'


# %% load dataset (to device)
composition = ToDevice(DEVICE)
dataset = InMemoryDataset(transform= composition)
dataset.load(osp.join(ROOT_FOLDER, '20240115-105053', 'processed', 'data.pt'))
# infer node labels from graph labels, required by embedding optimization
nodeLabels= torch.cat([g.y*torch.ones([g.num_nodes], device=DEVICE) 
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
        loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
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
        max_iter=150,
    )
    return acc


def embeddingLoop(dataset:Data, idxTrain, idxTest)-> torch.tensor:
    # define embedding model
    embedModel = Node2Vec(dataset.edge_index, **embedParams).to(DEVICE)
    # TODO consider increasing num_workers in model.loader
    embedLoader= embedModel.loader(batch_size=32, shuffle=True, num_workers=5)
    embedOptimizer= torch.optim.SparseAdam(list(embedModel.parameters()),lr=0.01)
    # nodeLabels are computed when loading the dataset
    
    epochs= tqdm(range(EMB_EPOCHS))
    for e in epochs:
        loss= embedTrain(embedModel, embedLoader, embedOptimizer)
        acc= embedTest(embedModel, nodeLabels, idxTrain, idxTest)
        epochs.set_description(f"EMB:{e}\t Lo:{loss:.4f}\t Ac:{acc:.4f}")
    
    
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
    def __init__(self, hdim=int):
        super().__init__()
        # random seed initialized glibal level for shuffling
        self.linear1 = torch.nn.Linear(embedParams['embedding_dim']*2, hdim)
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

class ReaderDataset(Dataset):
    """https://pytorch.org/docs/stable/data.html \n
    creates a dataset from computed tensor of embeddings
    """
    def __init__(self, embeddings:torch.Tensor, labels:torch.Tensor ):
        assert embeddings.shape[0] == labels.shape[0]
        self.x = embeddings.to(DEVICE)
        self.y = labels.reshape([-1]).to(DEVICE)
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


def learningLoop(hdim:int, randomSeed:int)-> torch.tensor:
    """
    performs training and return predicted values for best model
    """
    # set random seed
    randGen = torch.manual_seed(randomSeed)
    perm = torch.randperm(len(dataset), generator=randGen)
    b = int(len(dataset)*(1-TEST_SHARE)) 
    maskTrain, maskTest = perm[:b], perm[b:]
    # compute embeddings: dataset is a GLOBAL variable
    zz = embeddingLoop(dataset, maskTrain, maskTest)
    # clone and detach embeddings tensor to avoid autograd backtrace
    data = ReaderDataset(zz, dataset.y)
    # define NN model
    model= MLP(hdim).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    epochs= tqdm(range(MLP_EPOCHS))
    accWindow = torch.zeros([MA_WINDOW])
    for e in epochs:
        # FIXME  problem on second run of the loop
        loss = trainLoss(model, optimizer, criterion, data[maskTrain])
        acc = testAccuracy(model, data[maskTest])
        epochs.set_description(f"MLP:{e}\t Lo:{loss:.4f}\t Ac:{acc:.4f}")
        # early stopping with moving average update threshold
        accWindow = accWindow.roll(-1)
        accWindow[-1]=acc
        if (accWindow.max()-accWindow.min())<MA_THR:
            break
        
    
    # REMARK scores are not needed, computed later wirh information score
    return model(data.x).clone().detach()


# %% MAIN loop 
with open(osp.join(ROOT_FOLDER,'seedSequence.txt'),'r') as f:
    seeds = f.readlines()
    dtype=torch.long


seeds = [int(s) for s in seeds]

para = Parallel(n_jobs=1, return_as='generator')

outGen = para(delayed(learningLoop)(HIDDEN_DIM,s) for s in seeds)
#graph = dataset[0]# DEBUG 
outPreds = list(tqdm(outGen, descr='seed', total=len(seeds)))


# %% export clasification performance (to binaty .pt file)
result = torch.stack(outPreds).argmax(dim=2)
outVersion= dt.datetime.now().strftime("%Y%m%d-%H%M%S")
outPath= osp.join(ROOT_FOLDER,'embeddingML_')+outVersion
torch.save(result, outPath+'.pt')
# print training information
with open(outPath+'.log','w') as f :
    print(embedParams, file=f)
    print(f"exact epochs: {EMB_EPOCHS}", file=f)
    print(MLP(HIDDEN_DIM), file=f)
    print(f"max epochs: {MLP_EPOCHS}, moving average: {MA_WINDOW}", file=f)

