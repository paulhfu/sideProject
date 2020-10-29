import sys
import inspect
from torch_geometric.utils import degree
import torch
from torch_geometric.nn.conv import MessagePassing
import torch_scatter
from models.ril_function_models import NodeFeatureExtractor, EdgeFeatureExtractor
from models.simple_unet import DoubleConv
from collections import OrderedDict

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


def scatter_(name, src, index, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    if name == 'max':
        op = torch.finfo if torch.is_floating_point(src) else torch.iinfo
        fill_value = op(src.dtype).min
    else:
        fill_value = 0

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out


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

        node_features = self.message(*message_args)
        node_features = scatter_(self.aggr, node_features, edge_index[i], dim, dim_size=size[i])
        node_features = self.update(node_features, *update_args)
        return node_features

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


class EdgeConv(EdgeMessagePassing):
    def __init__(self, n_channels_in, n_channels_out, use_init_edge_feats=False, n_init_edge_channels=None,
                 n_hidden_layer=0, final_bn_nl=True):
        super(EdgeConv, self).__init__(aggr='no_aggr')  # no need for aggregation when only updating edges.

        m = 2
        hli = [torch.nn.Linear(n_channels_in * m, n_channels_in * (m + 2))]
        hli.append(torch.nn.LeakyReLU())
        # hli.append(torch.nn.BatchNorm1d(n_channels_in * (m + 2), track_running_stats=False))
        m += 2
        for i in range(n_hidden_layer):
            hli.append(torch.nn.Linear(n_channels_in * m, n_channels_in * (m + 2)))
            hli.append(torch.nn.LeakyReLU())
            # hli.append(torch.nn.BatchNorm1d(n_channels_in * (m + 2), track_running_stats=False))
            m += 2
        if use_init_edge_feats:
            hli.append(torch.nn.Linear(n_channels_in * m, n_channels_in * (m + 2)))
            # hli.append(torch.nn.LeakyReLU())
            # hli.append(torch.nn.BatchNorm1d(n_channels_in * (m + 2), track_running_stats=False))
        else:
            hli.append(torch.nn.Linear(n_channels_in * m, n_channels_out))
            # if final_bn_nl:
            #     hli.append(torch.nn.LeakyReLU())
            #     hli.append(torch.nn.BatchNorm1d(n_channels_out, track_running_stats=False))

        self.lin_edges_inner = torch.nn.Sequential(OrderedDict([("hl" + str(i), l) for i, l in enumerate(hli)]))
        if use_init_edge_feats:
            hlo = [(torch.nn.Linear(n_init_edge_channels + (n_channels_in * (m + 2)), n_channels_in * m))]
            hlo.append(torch.nn.LeakyReLU())
            hli.append(torch.nn.BatchNorm1d(n_channels_in * m, track_running_stats=False))
            for i in range(n_hidden_layer):
                hlo.append(torch.nn.Linear(n_channels_in * m, n_channels_in * (m-2)))
                hlo.append(torch.nn.LeakyReLU())
                # hlo.append(torch.nn.BatchNorm1d(n_channels_in * (m-2), track_running_stats=False))
                m -= 2
            hlo.append(torch.nn.Linear(n_channels_in * m, n_channels_out))
            # if final_bn_nl:
            #     hlo.append(torch.nn.LeakyReLU())
                # hlo.append(torch.nn.BatchNorm1d(n_channels_out, track_running_stats=False))

            self.lin_edges_outer = torch.nn.Sequential(OrderedDict([("hl"+str(i), l) for i, l in enumerate(hlo)]))


    def forward(self, x, edge_index, edge_features=None):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              edge_features=edge_features)

    def message(self, x_i, x_j, edge_index, edge_features):
        edge_sep = edge_index.shape[-1] // 2
        e_new = self.lin_edges_inner(torch.cat((x_i, x_j), dim=-1))
        loss_sym_edges = (((e_new[:edge_sep] - e_new[edge_sep:]) ** 2) / 2).mean()
        e_new = (e_new[:edge_sep] + e_new[edge_sep:]) / 2
        if edge_features is not None:
            e_new = self.lin_edges_outer(torch.cat((e_new, edge_features), dim=-1))
        return e_new, loss_sym_edges

class EdgeConvNoNodes(EdgeMessagePassing):
    def __init__(self):
        """
        expecting graph with only edges and returns node features as mean of edge features
        :param n_channels_out:
        :param n_node_channels:
        :param n_channels_interm:
        :param use_init_edge_feats:
        :param n_init_edge_channels:
        :param n_hidden_layer:
        """
        super(EdgeConvNoNodes, self).__init__(aggr='mean')  # no need for aggregation when only updating edges.

    def forward(self, edge_index, edge_features):
        edge_features = torch.cat([edge_features, edge_features], 0)
        return self.propagate(edge_index, size=(edge_index.max()+1, edge_index.max()+1), edge_features=edge_features)

    def message(self, edge_features):
        return edge_features

class NodeConv(EdgeMessagePassing):
    def __init__(self, n_channels_in, n_channels_out, n_hidden_layer=0, final_bn_nl=True):
        super(NodeConv, self).__init__(aggr='mean')  # no need for aggregation when only updating edges.

        m = 2
        hli = []
        for i in range(n_hidden_layer):
            hli.append(torch.nn.Linear(n_channels_in * m, n_channels_in * (m + 2)))
            hli.append(torch.nn.LeakyReLU())
            # hli.append(torch.nn.BatchNorm1d(n_channels_in * (m + 2), track_running_stats=False))
            m += 2
        hli.append(torch.nn.Linear(n_channels_in * m, n_channels_in * (m + 2)))
        hli.append(torch.nn.LeakyReLU())
        # hli.append(torch.nn.BatchNorm1d(n_channels_in * (m + 2), track_running_stats=False))

        self.lin_inner = torch.nn.Sequential(OrderedDict([("hl"+str(i), l) for i, l in enumerate(hli)]))

        hlo = [torch.nn.Linear(n_channels_in + (n_channels_in * (m + 2)), n_channels_in * m)]
        hli.append(torch.nn.LeakyReLU())
        for i in range(n_hidden_layer):
            hlo.append(torch.nn.Linear(n_channels_in * m, n_channels_in * (m - 2)))
            hlo.append(torch.nn.LeakyReLU())
            # hlo.append(torch.nn.BatchNorm1d(n_channels_in * (m - 2), track_running_stats=False))
            m -= 2
        hlo.append(torch.nn.Linear(n_channels_in * m, n_channels_out))
        # if final_bn_nl:
        #     hlo.append(torch.nn.LeakyReLU())
        #     hlo.append(torch.nn.BatchNorm1d(n_channels_out, track_running_stats=False))

        self.lin_outer = torch.nn.Sequential(OrderedDict([("hl"+str(i), l) for i, l in enumerate(hlo)]))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        edge_conv = self.lin_inner(torch.cat((x_i, x_j), dim=-1))
        return edge_conv

    def update(self, aggr_out, x):
        return self.lin_outer(torch.cat((x, aggr_out), dim=-1))
