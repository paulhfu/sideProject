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
    
    nodes = [edges.max() + 1 for edges in graphs]
    sizes = [edges.shape[0] * 2 for edges in graphs]
    all_edges = np.concatenate([edges.ravel() for edges in graphs], axis=0)
    # if len(edges) == 1:
    #     all_edges = all_edges[np.newaxis, ...]

    ccs, n_cs, sep_ccs = find_dense_subgraphs_impl(all_edges, k, nodes, sizes)
    ccs = ccs.reshape(-1, 2)
    sep_ccs = sep_ccs.reshape(-1, 2)
    sgs, sep_sgs = [], []
    _nc = 0
    bs = len(graphs)
    for scale_it in range(len(k)):
        nc = int(sum(n_cs[scale_it * bs: (scale_it+1) * bs])) * k[scale_it] + _nc

        _ccs = ccs[_nc:nc].reshape(-1, k[scale_it], 2)
        _sep_ccs = sep_ccs[_nc:nc].reshape(-1, k[scale_it], 2)

        _nc = nc
        szs = [0]
        for s in n_cs[scale_it * bs: (scale_it+1) * bs]:
            szs.append(int(s + szs[-1]))
            sg = _ccs[szs[-2]:szs[-1]]
            ssg = _sep_ccs[szs[-2]:szs[-1]]
            sgs.append(sg)
            sep_sgs.append(ssg)
    return sgs, sep_sgs
