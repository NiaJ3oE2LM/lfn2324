#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:19:54 2023
characterize node statistics and generate random graphs for DHFR
@author: hari
"""

import torch
from torch import sparse_coo_tensor
from torch.nn.functional import normalize

# dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data # graph object

# utility
from torch_geometric.utils import to_undirected, to_networkx
from utility import selectNodeFeats # fix poor one-hot encoding

# graphics
import networkx as nx
from matplotlib import pyplot as plt

#%% utility functions 
# TODO move to utility module with dependencies
def visualize_graph(graph:Data, color, title=''):
    fig, axs = plt.subplots(1,1)
    m = graph.num_edges
    if title:
        fig.suptitle(title+f" ({m} edges)")
    # network structure
    G = to_networkx(graph, to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42),
                     with_labels=True, node_size=100,
                     node_color=color, cmap="Set2", ax=axs)
    # adjacency matrix is not meaningful (permutations)
    fig.show()


# %% load dataset and fix poor encoding
dataAlias = 'DHFR'
transform= selectNodeFeats(torch.tensor([ 0,  5,  6,  7,  8, 15, 16, 34, 52]))
dataset = TUDataset(root='tmp', name=dataAlias, transform= transform)
# x shape is not updated BUT when you access the data, the transform is applied


#%% label distributions
ig = 234
graph = dataset[ig]
""" computation of label distribution (discrete)
use the edge index to create two stacks of one-hot encoded node labels
from the node features matrix: let A be the one generated from all edge sources
and B the stack generated form the destination of the edges.
Then the distribution is a simple matrix multiplication.
"""
E0 = graph.x[graph.edge_index[0,:]] 
E1 = graph.x[graph.edge_index[1,:]] 
labelDistribution =  E0.t()@E1 # the result is symmetric
# create corresponding probability matrix for each node based on the label
Ldist = graph.x @ labelDistribution @ graph.x.t()
# enforce null probability for self loops AND  normalize 
Pr_NL = normalize(Ldist-torch.diag(Ldist.diag()), dim=1)
assert torch.all(Pr_NL>=0) and torch.all(Pr_NL<=1)

#%% node degree probability (Chung-Lu) UNUSED

m = graph.num_edges
visualize_graph(graph, '#d0d0d0', f"DHFR {ig}")
# adjacency matrix (sparse)
A = sparse_coo_tensor(graph.edge_index, torch.ones(m))
# compute degrees (convert to dense afterwards, needed for outer product)
degs= A.sum(dim=0).to_dense()
# Chung-Lu probabilities: deg_u*deg_v (con be writen as outer product)
Pdist = torch.outer(degs,degs)
# remove self loops
Pdist = Pdist - torch.diag(Pdist.diag())
# FIXME normalize Chung-Lu formula
Pr_CL = Pdist / (2*m)
assert torch.all(Pr_CL>=0) and torch.all(Pr_CL<=1)

#%% generate random edges as sample of two independent events
# realization from Chung-Lu probability
# edges_CL = torch.bernoulli(Pr_CL)
# realizations from node label analysis
edges_NL = torch.bernoulli(Pr_NL**2) # FIXME why need a square ?
# combine indepndent events 
edges_sample = edges_NL #* edges_CL # removed after discussion 231214
# endure symmetry (undirected graph)
sample_index = edges_sample.to_sparse_coo().indices()
undir_index = to_undirected(sample_index)
# build sample graph object
graph_sample = Data(x=graph.x, edge_index=undir_index )


#%% Monte Carlo rando generation until criterias are met by chance
while graph_sample.has_isolated_nodes():
    edges_NL = torch.bernoulli(Pr_NL)
    edges_sample = edges_NL
    sample_index = edges_sample.to_sparse_coo().indices()
    undir_index = to_undirected(sample_index)
    graph_sample = Data(x=graph.x, edge_index=undir_index )


#%% sanity check
""" idea
because of the encoding, there are al lot of node labels that never appear.
these sould not be present in any random sample
"""
# self loops should not be present
assert not graph_sample.has_self_loops()
# there should be only one connected component
assert not graph_sample.has_isolated_nodes()
# TODO other ?
# check result visually
A_sample = sparse_coo_tensor(undir_index, torch.ones(undir_index.shape[1]))
visualize_graph(graph_sample, '#a0d0a0', 'MC Chung-Lu')