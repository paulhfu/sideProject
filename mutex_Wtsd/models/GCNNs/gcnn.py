from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import _pca_project, _pca_project_1d, plt_bar_plot
from collections import OrderedDict
from models.GCNNs.cstm_message_passing import NodeConv, EdgeConv, EdgeConvNoNodes


class Gcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, device, writer=None, final_bn_nl=False):
        super(Gcnn, self).__init__()
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv(n_in_channels, n_in_channels, n_hidden_layer=4)
        self.edge_conv1 = EdgeConv(n_in_channels, n_in_channels * 2, use_init_edge_feats=True, n_init_edge_channels=1,
                                    n_hidden_layer=4)
        self.node_conv2 = NodeConv(n_in_channels, n_in_channels, n_hidden_layer=4)
        self.edge_conv2 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True,
                                    n_init_edge_channels=n_in_channels * 2, n_hidden_layer=4, final_bn_nl=final_bn_nl)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input=False):

        node_features = self.node_conv1(node_features, edge_index)
        edge_features, side_loss_1 = self.edge_conv1(node_features, edge_index, angles)

        node_features = self.node_conv2(node_features, edge_index)
        edge_features, side_loss_2 = self.edge_conv2(node_features, edge_index, edge_features)

        side_loss = (side_loss_1 + side_loss_2) / 2

        if self.writer is not None and post_input and edge_features.shape[1] > 3:
            # with torch.no_grad():
            #     points = self.mean_shift(embeddings.reshape([-1, embeddings.shape[0]]))
            #     points = points.reshape(embeddings.shape)
            # plt.clf()
            # fig = plt.figure(frameon=False)
            # plt.imshow(_pca_project(points.detach().squeeze().cpu().numpy()), cmap='hot')
            # plt.colorbar()
            # self.writer.add_figure("image/embedding_ms_proj", fig, self.writer_counter)

            plt.clf()
            pca_proj_edge_fe = _pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PC1', 'PC2', 'PC3'])
            self.writer.add_figure("bar/embedding_proj_edge_features", fig, self.writer_counter)
            self.writer_counter += 1

        return edge_features, side_loss


class QGcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, device, writer=None, final_bn_nl=True):
        super(QGcnn, self).__init__()
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv(n_in_channels, n_in_channels, n_hidden_layer=4)
        self.edge_conv1 = EdgeConv(n_in_channels, n_in_channels * 2, use_init_edge_feats=True,
                                    n_init_edge_channels=2, n_hidden_layer=4)
        self.node_conv2 = NodeConv(n_in_channels, n_in_channels, n_hidden_layer=4)
        self.edge_conv2 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True,
                                    n_init_edge_channels=n_in_channels * 2, n_hidden_layer=4, final_bn_nl=final_bn_nl)

    def forward(self, node_features, edge_index, angles, gt_edges, actions, post_input=False):

        node_features = self.node_conv1(node_features, edge_index)
        edge_features, side_loss_1 = self.edge_conv1(node_features, edge_index, torch.cat([actions, angles], dim=-1))

        node_features = self.node_conv2(node_features, edge_index)
        edge_features, side_loss_2 = self.edge_conv2(node_features, edge_index, edge_features)

        side_loss = (side_loss_1 + side_loss_2) / 2

        if self.writer is not None and post_input and edge_features.shape[1] > 3:
            pca_proj_edge_fe = _pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PC1', 'PC2', 'PC3'])
            self.writer.add_figure("bar/embedding_proj_edge_features", fig, self.writer_counter)
            self.writer_counter += 1

        return edge_features, side_loss


class GlobalEdgeGcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_conv_its, device, writer=None, final_bn_nl=True):
        super(GlobalEdgeGcnn, self).__init__()
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.init_conv = EdgeConvNoNodes()
        self.node_conv = []
        for i in range(n_conv_its):
            self.node_conv.append(NodeConv(n_in_channels, n_in_channels, n_hidden_layer=0))
            super(GlobalEdgeGcnn, self).add_module(f"node_conv_{i}", self.node_conv[-1])

        self.edge_conv = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=False, n_hidden_layer=0,
                                   final_bn_nl=final_bn_nl)

    def forward(self, edge_features, edge_index):

        node_features = self.init_conv(edge_index, edge_features)
        for conv in self.node_conv:
            node_features = conv(node_features, edge_index)
        edge_features, side_loss = self.edge_conv(node_features, edge_index)

        return edge_features, side_loss

