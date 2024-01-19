# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:15:02 2024
utility module that containst useful functions and classes
@author: hari
"""

# transform requirements
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# feture computation
from torch import ones, sparse_coo_tensor, cat, tensor, where
from torch_geometric.utils import to_networkx
from networkx import (closeness_centrality, betweenness_centrality,
                      clustering)

# graphics
from networkx import draw_networkx, spring_layout
from matplotlib import pyplot as plt

# debug
from torch_geometric.datasets import TUDataset

# %% utility transforms

class addNodeFeats(BaseTransform):
    """
    callable function that concatenates the given feature(s) into the existing
    data node features. Asserts that the number of entries matches the num_nodes
    """
    def __init__(self, newNodeFeats):
        self.newNodeFeats= newNodeFeats # parameters you need
        
    def __call__(self, data: Data) -> Data:
        oldNodeFeats = data.x
        # assert dimension consistency and number of nodes
        assert len(oldNodeFeats.shape) ==  len(self.newNodeFeats.shape)
        assert oldNodeFeats.shape[0] == self.newNodeFeats.shape[0]
        # concatenate tensor along feature dimension
        data.x = cat((oldNodeFeats, self.newNodeFeats), dim=1)
        return data

# NOTE that if the transformation of the dataset is done after loading
# the dataset properties are no longer up to date


class selectNodeFeats(BaseTransform):
    """
    callable function that removes the node features corresponding to the 
    complement of th selected indeces. This comes handy when one-hot encoding
    is done poorly
    """
    def __init__(self, idx):
        self.idx= idx # indeces tensor
        
    def __call__(self, data: Data) -> Data:
        d = data.x.shape[1]
        # assert indeces consistency
        assert d > self.idx.max()
        assert self.idx.min() >=0
        # select features using given indeces
        data.x = data.x[:,self.idx]
        return data

# NOTE that if the transformation of the dataset is done after loading
# the dataset properties are no longer up to date


class onehotDecode(BaseTransform):
    """
    replaces one-hot encoded features with the corresponding integer
    output data will have num_feats=1
    """
    def __init__(self, start:int, labels:tensor):
        # TODO preserve before and after
        assert start >=0
        self.start = start
        self.labels= labels # ordered labels
        self.end= start + len(labels)
        
    def __call__(self, graph: Data) -> Data:
        n, d = graph.x.shape
        # assert indeces consistency
        assert d >= self.end 
        # preserve features (afterFeats not working)
        beforeFeats= graph.x[:,0:self.start]
        encFeats = graph.x[:,self.start:self.end]
        afterFeats= graph.x[:, self.end:]
        # decode features
        labelStack = self.labels.repeat([n,1])
        decodeFeats= labelStack[where(encFeats>0)].reshape([n,1])
        graph.x = cat((beforeFeats,decodeFeats,afterFeats), dim=1)
        return graph


# %% analytical transforms

class nodeDegree(BaseTransform):
    """
    append NODE DEGREE to the current feature matrix
    the node degree centrality is normalized node degree in NetworkX
    """
    def __init__(self):
        return None
        
    def __call__(self, graph: Data) -> Data:
        oldNodeFeats = graph.x
        # compute feature
        m = graph.num_edges
        # adjacency matrix (sparse)
        A = sparse_coo_tensor(graph.edge_index, ones(m))
        # compute degrees (convert to dense afterwards, needed for outer product)
        degs= A.sum(dim=0).to_dense().reshape([-1,1])
        graph.x = cat((oldNodeFeats, degs), dim=1)
        return graph

# TODO if keep using networkx consider using nx-cugraph backend fro GPU accel

class closenessCentrality(BaseTransform):
    """
    append CLOSENESS CENTRALITY to the current feature matrix
    uses NetworkX function closeness_centrality
    """
    def __init__(self, undirected : bool = True):
        self.undirected = undirected # to_networkx
        
    def __call__(self, graph: Data) -> Data:
        oldNodeFeats = graph.x
        # compute feature with nx
        nx_graph = to_networkx(graph,to_undirected=self.undirected)
        feat = closeness_centrality(nx_graph).values()
        out =  tensor(list(feat)).reshape([-1,1])
        graph.x = cat((oldNodeFeats, out), dim=1)
        return graph


class betweennessCentrality(BaseTransform):
    """
    append BETWEENNESS CENTRALITY to the current feature matrix
    uses NetworkX function betweenness_centrality (cugraph possible)
    """
    def __init__(self, undirected : bool = True):
        self.undirected = undirected # to_networkx
        
    def __call__(self, graph: Data) -> Data:
        oldNodeFeats = graph.x
        # compute feature with nx
        nx_graph = to_networkx(graph,to_undirected=self.undirected)
        feat = betweenness_centrality(nx_graph).values()
        out =  tensor(list(feat)).reshape([-1,1])
        graph.x = cat((oldNodeFeats, out), dim=1)
        return graph
    

class nodeClustering(BaseTransform):
    """
    append NODE CLUSTERING to the current feature matrix
    uses NetworkX function clustering
    """
    def __init__(self, undirected : bool = True):
        self.undirected = undirected # to_networkx
        
    def __call__(self, graph: Data) -> Data:
        oldNodeFeats = graph.x
        # compute feature with nx
        nx_graph = to_networkx(graph,to_undirected=self.undirected)
        feat = clustering(nx_graph).values()
        out =  tensor(list(feat)).reshape([-1,1])
        graph.x = cat((oldNodeFeats, out), dim=1)
        return graph
    

# TODO implement other features ? -> check lecture notes

# TODO append graph level metrics to graph labels: motifs
# https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.clustering.motifs.html#graph_tool.clustering.motifs


# %% visual inspection

def visualize_graph(graph:Data, color, title=''):
    fig, axs = plt.subplots(1,1)
    m = graph.num_edges
    if title:
        fig.suptitle(title+f" ({m} edges)")
    # network structure
    G = to_networkx(graph, to_undirected=True)
    draw_networkx(G, pos=spring_layout(G, seed=42),
                     with_labels=True, node_size=100,
                     node_color=color, cmap="Set2", ax=axs)
    # adjacency matrix is not meaningful (permutations)
    fig.show()
    
# TODO compare sample against oiginal (proba distributions)

# %% DEBUG 
if __name__ == '__main__':
    dataAlias = 'DHFR'
    composition = nodeDegree()
    dataset = TUDataset(root='tmp', name=dataAlias, use_node_attr=True,
                        transform= composition)
    labels = tensor(list(range(53)))
    trans = onehotDecode(start=3, labels=labels)
    oldGraph=  dataset[345]
    newGraph= trans(oldGraph)
