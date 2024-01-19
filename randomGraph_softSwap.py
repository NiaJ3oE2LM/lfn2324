# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:14:41 2024
generate random graphs with probabilistic fixed degree swapping
1. nodes with higher degree are more likely to swap (Chung-Lu)
2. compliant swapping is sampled from node label distribution 
@author: hari
"""

from torch import where, tensor, ones, Size
from torch import outer, diag, sparse_coo_tensor , all
from torch.nn.functional import normalize
from torch.distributions.multinomial import Multinomial
from torch.distributions import Bernoulli

# dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, Data

# utility
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix
from utility import selectNodeFeats, visualize_graph # fix poor one-hot encoding
from scipy.sparse.csgraph import connected_components

# job handling
#from multiprocessing import Pool # exceeds ulimit 
from joblib import Parallel, delayed 
import datetime as dt
import os.path as osp
from tqdm import tqdm
import argparse

# %% load dataset and fix poor encoding
dataAlias = 'DHFR'
transform= selectNodeFeats(tensor([ 0,  5,  6,  7,  8, 15, 16, 34, 52]))
dataset = TUDataset(root='tmp', name=dataAlias, transform= transform)
# x shape is not updated BUT when you access the data, the transform is applied


# %% main function for parallelization

def randomGraph(graph:Data)-> Data:
    #  %%% hyperameters 
    swap_todo= graph.num_edges*2
    
    #  %%% node labels probabilities (for the graph)
    #visualize_graph(graph, '#d0d0d0', f"DHFR {ig}")
    E0 = graph.x[graph.edge_index[0,:]] 
    E1 = graph.x[graph.edge_index[1,:]] 
    labelDistribution =  E0.t()@E1 # the result is symmetric
    # create corresponding probability matrix for each node based on the label
    Ldist = graph.x @ labelDistribution @ graph.x.t()
    # enforce null probability for self loops AND  normalize 
    Pr_NL = normalize(Ldist-diag(Ldist.diag()), dim=1)
    assert all(Pr_NL>=0) and all(Pr_NL<=1)

    #  %%% node degree probability (Chung-Lu)
    m = graph.num_edges
    # adjacency matrix (sparse)
    A = sparse_coo_tensor(graph.edge_index, ones(m))
    # compute degrees (convert to dense afterwards, needed for outer product)
    degs= A.sum(dim=0).to_dense()
    # Chung-Lu probabilities: deg_u*deg_v (con be writen as outer product)
    Pdist = outer(degs,degs)
    # remove self loops
    Pdist = Pdist - diag(Pdist.diag())
    # normalize Chung-Lu formula
    Pr_CL = Pdist /(2*m)
    assert all(Pr_CL>=0) and all(Pr_CL<=1)

# TODO try node degree probability (label based)

    #  %%% probabilistic swapping with node labels
    swap_done= 0
    # associate degree-based distribution to graph edges
    #degs_edges = degs[graph.edge_index[0]] + degs[graph.edge_index[1]]
    degs_edges = Pr_CL[graph.edge_index[0],graph.edge_index[1]]
    while swap_done < swap_todo:
        # sample two edges
        sample = Multinomial(2,degs_edges).sample()
        jj = where(sample>0)[0]
        try:  # ensure two edges
            assert jj.shape == Size([2])
        except: # bad lucj, less than 2 edges
            continue
        u, v = graph.edge_index[:,jj][:,0]
        z, w = graph.edge_index[:,jj][:,1]
        # ensure different edges (undirected graph)
        diff_ok = u!=w and v!=z
        # check if swapped edges are new
        tmp = graph.edge_index[:,where(graph.edge_index[0] == u)[0]]
        uw_ok = where(tmp[1]==w)[0].shape == Size([0])
        tmp = graph.edge_index[:,where(graph.edge_index[0] == z)[0]]
        zv_ok = where(tmp[1]==v)[0].shape == Size([0])
        # ensure one connected component: do not swap nodes all with degree 1
        degs_ok = degs[u]*degs[w]>1 and degs[z]*degs[v]>1 
        if all(tensor([diff_ok, uw_ok, zv_ok, degs_ok])):
            # sample the corresponding node label distributions
            uw_sample = Bernoulli(Pr_NL[u,w]).sample()==tensor(1)
            zv_sample = Bernoulli(Pr_NL[z,v]).sample()==tensor(1)
            # if positive, implement the swap (undirected)
            if uw_sample and zv_sample:
                #print(f"u({u})-v({v}) <> z({z})-w({w})") # DEBUG
                """ graph is undirected (swap operation)
                id:  j j     i i      =>       j j     i i
                [... u z ... v w ...      [... u z ...(w v)... 
                 ... v w ... u z ...]      ...(w v)... u z ...]
                """
                # find index i for edge v-u
                tmp = where(graph.edge_index[0]==v)[0]
                i = tmp[0] + where(graph.edge_index[:,tmp][1]==u)[0]
                # find index i for edge w-z
                tmp = where(graph.edge_index[0]==w)[0]
                ii = tensor([i, tmp[0] + where(graph.edge_index[:,tmp][1]==z)[0]])
                # copy edge_index and modify entries
                edges = graph.edge_index.clone().detach()
                edges[:,jj]= tensor([[u,z],[w,v]])
                edges[:,ii]= tensor([[w,v],[u,z]])
                try: # reject the modification if resulting in cc>1
                    adj = to_scipy_sparse_matrix(edges)
                    cc, _ = connected_components(adj, directed=False)
                    assert cc==1
                except:
                    continue
                # update graph object with modified edges
                # FIXME to_undirected casting should NOT be required 
                graph.update({'edge_index': to_undirected(edges)})
                swap_done += 1
    
    
    #  %%% sanity check
    # TODO move sanity check on stored random collection   
    assert graph.is_undirected()
    # self loops should not be present
    assert not graph.has_self_loops()
    # there should be only one connected component
    assert not graph.has_isolated_nodes()
    # one connected component ensured above
    # node degree should have been preserved
    A_new = sparse_coo_tensor(graph.edge_index, ones(m))
    degs_new= A_new.sum(dim=0).to_dense()
    assert all(degs == degs_new)
    # visual inspection
    #visualize_graph(graph, '#a0d0a0', 'RW')

    return graph


# %%  create in memory dataset

class storeDataset(InMemoryDataset):
    """
    Allows to store on disk a huge list of Data objects (resulting from
    the random computation) through PyG standard formalism.
    In order to load data from disk, initialize an empty InMemoryDataset
    and call the load method directly on the file root/processed/data.pt
    Feature computation on random graph is delayed to later processing.
    """
    def __init__(self, root, importData):
        self.list = importData
        # NO transform, pre_filter, pre_transform
        super().__init__(root, None, None, None)
        # using PyG 2.4.0:
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [] # list fed as parameter
        
    @property
    def processed_file_names(self):
        return ['data.pt'] # no pre_filter nor pre_transform

    def download(self):
        return None # not needed, leave empty

    def process(self):      
        self.save(self.list, self.processed_paths[0])


# %% run jobs in parallel and store result
parser = argparse.ArgumentParser(description='DHFR random generator')
parser.add_argument('num_col', metavar='N', type=int,
                help='number of collections to generate')
    
if __name__ == '__main__':
    args = parser.parse_args()
    # TODO make argument 
    rootFolder= 'random' # InMemoryDataset creates the folder is needed
    para = Parallel(n_jobs=10, return_as='generator')
    for i in tqdm(range(args.num_col), desc='collection'): # about 15 mins/coll @ 10 proc.
        outGen = para(delayed(randomGraph)(g) for g in dataset)
        #graph = dataset[0]# DEBUG 
        randCollection = list(tqdm(outGen, total=len(dataset)))
        
        # store result as PyG Dataset
        outName= dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        rootPath= osp.join(rootFolder,outName)
        storeDataset(rootPath, randCollection)
