#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:39:26 2023
compute local clustering coefficient for DHFR(-MD) dataset (TUD)
@author: hari
"""
from multiprocessing import Pool # TODO implement 

import torch
# geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
# sparse matrix implementation for adjacency matrix
from torch import sparse_coo_tensor


#%% load dataset 

dataAlias = 'DHFR'
dataset = TUDataset(root="/".join(['.','tmp']), name=dataAlias)

#%% local clustering coefficient for all graphs (UNDIRECTED)

def LCCs(graph:Data)->Data:
    """
    Exactly computes the Local Clustering Coeffient (LCC) for each node in 
    the graph and stacks the values as a new feature at the end of the 
    feature matrix.
    See lecture [08] for details.

    Parameters
    ----------
    graph : Data (Pytorch Geometric data structure for graph)
        DESCRIPTION.

    Returns
    -------
    Data (Pytorch Geometric data structure for graph)
        DESCRIPTION.

    """
    ee = graph.edge_index # list of edges
    m = graph.edge_index.shape[-1]//2 # number of edges
    # set of nodes (with at least one edge)
    vv = ee[0,:].unique()
    # sparse adjacency matrix
    E_sparse = sparse_coo_tensor(graph.edge_index, torch.ones(2*m))
    # concatenate local clustering coefficient as new feature
    graph.x = torch.cat([graph.x, torch.zeros((graph.x.shape[0],1))], dim=1)
    for v in vv:
        # compute neighbor of v
        N_v = ee[1,torch.where(ee[0,:]==v)[0]]
        for u1 in N_v:
            for u2 in N_v:
                if E_sparse[u1,u2]:
                    graph.x[v,-1] = +1 # increlement counter (last feature)
    
    # TODO normalization factor deg(deg-1) required for computing LCC
    
    return graph
    

LCCs(dataset[2])
#Pool(10).map(LCCs, dataset)

#%% visualize graph molecule
""" Introduction: Hands-on Graph Neural Networks
https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=zF5bw3m9UrMy
"""

# Helper function for visualization.
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()
    # TODO add edge labels for comparison with 

from torch_geometric.utils import to_networkx
from random import randrange

#%% inspect LCC at random

for i in range(2):
    g = randrange(0,len(dataset),1)
    mol = dataset[g]
    G = to_networkx(mol, to_undirected=True)
    visualize_graph(G, color=mol.x[:,-1])
    plt.title(g)
    # check LCC with another implementation (networkx)
    lcc = nx.clustering(G)
    print(f"{g}:\t{list(lcc.values())}")


#%% save computed  