#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:01:12 2024
generate random graphs by computing node label disributions and random walks
Ensure each sample is connected by overlapping multiple random walks
Check the resulting node degree for similarity with respect to original sample
@author: hari
"""
from torch import where, tensor, ones, zeros, cat
from torch import outer, diag, sparse_coo_tensor , all
from torch import Tensor, Size
from torch.nn.functional import normalize
from torch.distributions.multinomial import Multinomial

# dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data # graph object
#from torch_geometric.transforms import LargestConnectedComponents

from scipy.sparse.csgraph import connected_components

# utility
from torch_geometric.utils import to_undirected, to_networkx, to_scipy_sparse_matrix
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


def runRandWalks(number:int, length:int,
                 Pr:Tensor, degs_ref:Tensor, degs_now:Tensor)  :
    # TODO check that Pr dimension is geq than number of degs
    assert degs_ref.shape == degs_now.shape
    # walk start from uncon. nodes with degree different from original
    Pr_start = degs_ref-degs_now
    Pr_start[where(Pr_start<0)]=0 # stop degree overflow
    i = where(Multinomial(number, Pr_start).sample()>0)[0]
    # init random walk
    sample = Multinomial(1,Pr[i,:]).sample()
    _, j = where(sample>0)
    path = cat((i.reshape([1,-1]),j.reshape([1,-1])), dim=0)
    # cannot walk backwards
    batch_inv = cat((j.reshape([1,-1]),i.reshape([1,-1])), dim=0)
    Pr[tuple(path)]=1e-3
    Pr[tuple(batch_inv)]=1e-3
    # complete the rw short of one step
    for k in range(len_rw-1): 
        # next step in the walk
        i=j.clone().detach()
        # rw step
        sample = Multinomial(1,Pr[i,:]).sample()
        _, j = where(sample>0)
        batch = cat((i.reshape([1,-1]),j.reshape([1,-1])), dim=0)
        path = cat((path, batch),dim=1)
        # cannot walk backwards
        batch_inv = cat((j.reshape([1,-1]),i.reshape([1,-1])), dim=0)
        Pr[tuple(batch)]=1e-3
        Pr[tuple(batch_inv)]=1e-3
    
    # sanity check
    try:
        assert path.shape == Size([2,number*length])
    except:
        print(path.shape)
    
    path_rw = path.clone().detach()
    # connected should have less proba to be chosen as starting point
    degs_now[path_rw.unique()] += 1
    degs_rw = degs_now.clone().detach()
    # return values for next iterations
    return path_rw, degs_rw


# %% load dataset and fix poor encoding
dataAlias = 'DHFR'
transform= selectNodeFeats(tensor([ 0,  5,  6,  7,  8, 15, 16, 34, 52]))
dataset = TUDataset(root='tmp', name=dataAlias, transform= transform)
# x shape is not updated BUT when you access the data, the transform is applied

# %% compute node labels probabilities (for a graph)
ig = 500
graph = dataset[ig]
visualize_graph(graph, '#d0d0d0', f"DHFR {ig}")
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
Pr_NL = normalize(Ldist-diag(Ldist.diag()), dim=1)
assert all(Pr_NL>=0) and all(Pr_NL<=1)

#%% node degree probability (Chung-Lu)

m = graph.num_edges
# adjacency matrix (sparse)
A = sparse_coo_tensor(graph.edge_index, ones(m))
# compute degrees (convert to dense afterwards, needed for outer product)
degs= A.sum(dim=0).to_dense()
# Chung-Lu probabilities: deg_u*deg_v (con be writen as outer product)
Pdist = outer(degs,degs)
# remove self loops
Pdist = Pdist - diag(Pdist.diag())
# FIXME normalize Chung-Lu formula
Pr_CL = normalize(Pdist, dim=1)
assert all(Pr_CL>=0) and all(Pr_CL<=1)


# %% generate new random sample with n rw of specified length
num_rw = graph.num_nodes* 1//2
len_rw = 1
# initialize graph (preserve node labels)
graph_sample = Data(x=graph.x)
# rw modifies the probability of visited nodes
Pr_rw = normalize(Pr_NL, dim=1).clone().detach()
# LOOP2 Monte Carlo to ensure exactly 1 connected component
cc_sample = graph_sample.num_nodes
while cc_sample>1:
    # init degs to zero
    degs_rw = zeros(degs.shape)
    # DEBUG fnction run rw
    number = num_rw; length = len_rw; Pr = Pr_rw.clone().detach(); degs_ref = degs.clone().detach(); degs_now= degs_rw.clone().detach()
    path_rw, degs_rw = runRandWalks(num_rw,len_rw,Pr_rw,degs,degs_rw)
    # add RW path to sto_scipy_sparse_matrixample graph
    graph_sample.update({'edge_index':path_rw})
    
    # LOOP1 until one connected component
    while graph_sample.has_isolated_nodes():
        # DEBUG fnction run rw
        number = num_rw; length = len_rw; Pr = Pr_rw.clone().detach(); degs_ref = degs.clone().detach(); degs_now= degs_rw.clone().detach() 
        path_rw, degs_rw = runRandWalks(num_rw,len_rw,Pr_rw,degs,degs_rw)
        # add RW path to sample graph
        new_edge_index = cat((graph_sample.edge_index, path_rw), dim=1)
        graph_sample.update({'edge_index':new_edge_index})
    
    # (LOOP2) assert connected components
    adj = to_scipy_sparse_matrix(graph_sample.edge_index)
    cc_sample, _ = connected_components(adj, directed=False)


# cast resulting sample to undirected graph 
undir_index = to_undirected(graph_sample.edge_index)
graph_sample = Data(x=graph.x, edge_index=undir_index)

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
A_sample = sparse_coo_tensor(undir_index, ones(undir_index.shape[1]))
visualize_graph(graph_sample, '#a0d0a0', 'RW')

