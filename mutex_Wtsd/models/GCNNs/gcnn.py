from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import _pca_project, _pca_project_1d, plt_bar_plot
from models.GCNNs.cstm_message_passing import NodeConv1, EdgeConv2


class Gcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, device, writer=None):
        super(Gcnn, self).__init__()
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv1(n_in_channels, n_in_channels, n_hidden_layer=5)
        self.edge_conv1 = EdgeConv2(n_in_channels, n_in_channels, 3 * n_in_channels, n_hidden_layer=5)
        self.node_conv2 = NodeConv1(n_in_channels, n_in_channels, n_hidden_layer=5)
        self.edge_conv2 = EdgeConv2(n_out_channels, n_in_channels, 3 * n_in_channels,
                                    use_init_edge_feats=True, n_init_edge_channels=3 * n_in_channels,
                                    n_hidden_layer=5)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input=False):

        node_features, _ = self.node_conv1(node_features, edge_index, angles)
        side_loss1, edge_features = self.edge_conv1(node_features, edge_index)

        node_features, _ = self.node_conv2(node_features, edge_index, angles)
        side_loss2, edge_features = self.edge_conv2(node_features, edge_index, edge_features)

        side_loss = (side_loss1 + side_loss2) / 2

        if self.writer is not None and post_input and edge_features.shape[1] > 3:
            # with torch.no_grad():
            #     points = self.mean_shift(embeddings.reshape([-1, embeddings.shape[0]]))
            #     points = points.reshape(embeddings.shape)
            # plt.clf()
            # fig = plt.figure(frameon=False)
            # plt.imshow(_pca_project(points.detach().squeeze().cpu().numpy()), cmap='hot')
            # plt.colorbar()
            # self.writer.add_figure("image/embedding_ms_proj", fig, self.writer_counter)
            pca_proj_edge_fe = _pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PCA1', 'PCA2', 'PCA3'])
            self.writer.add_figure("bar/embedding_proj_edge_features", fig, self.writer_counter)
            self.writer_counter += 1

        return edge_features, side_loss


class QGcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, device, writer=None):
        super(QGcnn, self).__init__()
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv1(n_in_channels, n_in_channels, n_hidden_layer=5)
        self.edge_conv1 = EdgeConv2(3 * n_in_channels, n_in_channels, 3 * n_in_channels, use_init_edge_feats=True,
                                    n_init_edge_channels=1, n_hidden_layer=5)
        self.node_conv2 = NodeConv1(n_in_channels, n_in_channels, n_hidden_layer=5)
        self.edge_conv2 = EdgeConv2(n_out_channels, n_in_channels, 3 * n_in_channels,
                                    use_init_edge_feats=True, n_init_edge_channels=3 * n_in_channels,
                                    n_hidden_layer=5)

    def forward(self, node_features, edge_index, angles, gt_edges, actions, post_input=False):

        node_features, _ = self.node_conv1(node_features, edge_index, angles)
        side_loss1, edge_features = self.edge_conv1(node_features, edge_index, actions)

        node_features, _ = self.node_conv2(node_features, edge_index, angles)
        side_loss2, edge_features = self.edge_conv2(node_features, edge_index, edge_features)

        side_loss = (side_loss1 + side_loss2) / 2

        if self.writer is not None and post_input and edge_features.shape[1] > 3:
            # with torch.no_grad():
            #     points = self.mean_shift(embeddings.reshape([-1, embeddings.shape[0]]))
            #     points = points.reshape(embeddings.shape)
            # plt.clf()
            # fig = plt.figure(frameon=False)
            # plt.imshow(_pca_project(points.detach().squeeze().cpu().numpy()), cmap='hot')
            # plt.colorbar()
            # self.writer.add_figure("image/embedding_ms_proj", fig, self.writer_counter)
            pca_proj_edge_fe = _pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PCA1', 'PCA2', 'PCA3'])
            self.writer.add_figure("bar/embedding_proj_edge_features", fig, self.writer_counter)
            self.writer_counter += 1

        return edge_features, side_loss