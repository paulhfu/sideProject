from ._rag_utils import find_dense_subgraphs_impl 
import numpy as np

def find_dense_subgraphs(graphs, k):
    """
    gets subgraphs sg_i in a graph with |E|=k s.t. Union_i(sg_i)=graph
    graphs is an iterable of graphs, where each element is ndarra of shape=(|E|, 2)

    constraints on each graph are:
      - is defined by edge list (list if ints)
      - it is connected
      - node defs are consecutive integers
    the subgraphs have a high density if the graph is a rag
    """
    for edges in graphs:
        assert all(np.unique(edges) == np.arange(edges.max()+1)), "graph constraints not satisfied"
    
    nodes = [edges.max() for edges in graphs]

    sizes = [graph.edges[0] for edges in graphs]

    all_edges = np.concatenate([edges.ravel() for edges in graphs], axis=0)

    ccs, n_cs = find_dense_subgraphs_impl(all_edges, k, nodes, sizes)

    ccs = ccs.reshape(-1, k, 2)
    szs = [0] 
    for s in n_cs:
        szs.append(s*2 + szs[-1])

    sgs = [ccs[szs[i]:szs[i+1]] for i in range(szs.size()-1)]

    return sgs


