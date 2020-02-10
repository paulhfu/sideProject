import sys
import inspect
from torch_geometric.utils import degree
import torch
from utils import check_no_singles
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter_

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


class EdgeMessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers
        copied from PyG and edited
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
    """
    def __init__(self, aggr='add', flow='source_to_target'):
        super(EdgeMessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', 'no_aggr']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, dim=0, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferred and assumed to be symmetric.
                (default: :obj:`None`)
            dim (int, optional): The axis along which to aggregate.
                (default: :obj:`0`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        dim = 0
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(dim)
                            if size[1 - idx] != tmp[1 - idx].size(dim):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if tmp is None:
                        message_args.append(tmp)
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(dim)
                        if size[idx] != tmp.size(dim):
                            raise ValueError(__size_error_msg__)

                        tmp = torch.index_select(tmp, dim, edge_index[idx])
                        message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        if self.aggr == "no_aggr":
            return self.message(*message_args)

        out, edge_features = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim, dim_size=size[i])
        out = self.update(out, *update_args)
        return out, edge_features

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


class GcnEdgeConv(EdgeMessagePassing):
    def __init__(self, n_node_features_in, n_edge_features_in, n_node_features_out, n_edge_features_out):
        super(GcnEdgeConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin_nodes_inner = torch.nn.Linear(n_node_features_in + n_edge_features_in, n_node_features_in)
        self.lin_edges_inner = torch.nn.Linear(n_node_features_in * 2, n_edge_features_in)
        self.lin_nodes_outer = torch.nn.Linear(n_node_features_in * 2, n_node_features_out)
        self.lin_edges_outer = torch.nn.Linear(n_edge_features_in * 2, n_edge_features_out)

    def forward(self, x, e, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e=e)

    def message(self, x_i, x_j, e, edge_index, size):
        edge_sep = edge_index.shape[-1]//2
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x_new = self.lin_nodes_inner(torch.cat((torch.cat((e, e)), x_j), dim=1))
        x_new = torch.cat((norm.view(-1, 1) * x_new, x_i), dim=1)
        e_new = self.lin_edges_inner(torch.cat((x_i, x_j), dim=1))
        e_new = (e_new[:edge_sep] + e_new[edge_sep:]) / 2
        e_new = self.lin_edges_outer(torch.cat((e_new, e), dim=1))
        return x_new, e_new

    def update(self, aggr_out):
        return self.lin_nodes_outer(aggr_out)


class NodeConv(MessagePassing):
    def __init__(self, n_node_features_in, n_node_features_out):
        super(NodeConv, self).__init__(aggr='mean')  # no need for aggregation when only updating edges.
        self.lin_inner = torch.nn.Linear(n_node_features_in , n_node_features_out // 2)
        self.lin_outer = torch.nn.Linear(n_node_features_out // 2 + n_node_features_in, n_node_features_out)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return self.lin_inner(x_j)

    def update(self, aggr_out, x):
        return self.lin_outer(torch.cat((aggr_out, x), dim=1))


class EdgeConvNoEdge(EdgeMessagePassing):
    def __init__(self, n_node_features_in, n_edge_features_out):
        super(EdgeConvNoEdge, self).__init__(aggr='no_aggr')  # no need for aggregation when only updating edges.
        self.lin_edges_inner = torch.nn.Linear(n_node_features_in * 2, n_node_features_in * 2)
        self.lin_edges_outer = torch.nn.Linear(n_node_features_in * 2, n_edge_features_out)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        edge_sep = edge_index.shape[-1] // 2
        e_new = self.lin_edges_inner(torch.cat((x_i, x_j), dim=1))
        e_new = (e_new[:edge_sep] + e_new[edge_sep:]) / 2
        e_new = self.lin_edges_outer(e_new)
        return e_new


class EdgeConv(EdgeMessagePassing):
    def __init__(self, n_node_features_in, n_edge_features_in, n_edge_features_out):
        super(EdgeConv, self).__init__(aggr='no_aggr')  # no need for aggregation when only updating edges.
        self.lin_edges_inner = torch.nn.Linear(n_node_features_in * 2, n_node_features_in * 2)
        self.lin_edges_outer = torch.nn.Linear(n_node_features_in * 2 + n_edge_features_in, n_edge_features_out)

    def forward(self, x, e, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e=e)

    def message(self, x_i, x_j, e, edge_index, size):
        edge_sep = edge_index.shape[-1] // 2
        e_new = self.lin_edges_inner(torch.cat((x_i, x_j), dim=1))
        e_new = (e_new[:edge_sep] + e_new[edge_sep:]) / 2
        e_new = self.lin_edges_outer(torch.cat((e_new, e), dim=1))
        return e_new
