import torch
from models.GCNNs.cstm_message_passing import NodeConv, EdgeConv
import torch.nn.functional as F
import torch.nn as nn
from models.sp_embed_unet import SpVecsUnet, SpVecsUnet
import matplotlib.pyplot as plt
from utils.general import _pca_project
# import gpushift


class GcnEdgeAngle1dQ(torch.nn.Module):
    def __init__(self, n_raw_channels, n_embedding_channels, n_edge_features_in, n_edge_classes, device, softmax=True,
                 writer=None):
        super(GcnEdgeAngle1dQ, self).__init__()
        self.writer = writer
        self.fe_ext = SpVecsUnet(n_raw_channels, n_embedding_channels, device)
        self.softmax = softmax
        n_embedding_channels += 1
        self.node_conv1 = NodeConv(n_embedding_channels, n_embedding_channels, n_hidden_layer=5)
        self.edge_conv1 = EdgeConv(n_embedding_channels, n_embedding_channels, 3 * n_embedding_channels, n_hidden_layer=5)
        self.node_conv2 = NodeConv(n_embedding_channels, n_embedding_channels, n_hidden_layer=5)
        self.edge_conv2 = EdgeConv(n_embedding_channels, n_embedding_channels, 3 * n_embedding_channels,
                                   use_init_edge_feats=True, n_init_edge_channels=3 * n_embedding_channels,
                                   n_hidden_layer=5)

        # self.lstm = nn.LSTMCell(n_embedding_channels + n_edge_features_in + 1, hidden_size)

        self.out_q = nn.Sequential(
            nn.Linear(n_embedding_channels + n_edge_features_in + 1, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 256),
            nn.Linear(256, n_edge_classes),
        )

        self.device = device
        self.writer_counter = 0

        # # note: run without %time first
        # self.mean_shift = gpushift.MeanShift(
        #     n_iter=40,
        #     kernel=gpushift.MeanShiftStep.GAUSSIAN_KERNEL,
        #     bandwidth=0.2,
        #     blurring=False,
        #     use_keops=True
        # )

    def forward(self, state, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None, round_n=None,
                post_input=False):
        edge_weights = state[0].to(self.device)
        input = state[2].unsqueeze(0).unsqueeze(0).to(self.device)

        # input = torch.stack((state[1], state[2], state[3])).unsqueeze(0).to(self.device)
        if sp_indices is None:
            return self.fe_ext(input)
        if edge_features_1d is None:
            return self.fe_ext(input, sp_indices)
        node_features, embeddings = self.fe_ext(input, sp_indices)
        embeddings = embeddings.squeeze()

        if self.writer is not None and post_input:
            # with torch.no_grad():
            #     points = self.mean_shift(embeddings.reshape([-1, embeddings.shape[0]]))
            #     points = points.reshape(embeddings.shape)
            # plt.clf()
            # fig = plt.figure(frameon=False)
            # plt.imshow(_pca_project(points.detach().squeeze().cpu().numpy()), cmap='hot')
            # plt.colorbar()
            # self.writer.add_figure("image/embedding_ms_proj", fig, self.writer_counter)
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(_pca_project(embeddings.detach().squeeze().cpu().numpy()), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(state[1].unsqueeze(0).cpu().squeeze(), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/state1", fig, self.writer_counter)
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(state[2].unsqueeze(0).cpu().squeeze(), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/state2", fig, self.writer_counter)
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(state[3].unsqueeze(0).cpu().squeeze(), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/state3", fig, self.writer_counter)
            self.writer_counter += 1

        node_features = torch.cat(
            [node_features, torch.ones([node_features.shape[0], 1], device=node_features.device) * round_n], -1)
        node_features, _ = self.node_conv1(node_features, edge_index, angles)
        side_loss1, edge_features = self.edge_conv1(node_features, edge_index, torch.cat((edge_weights, edge_weights), dim=0))

        node_features, _ = self.node_conv2(node_features, edge_index, angles)
        side_loss2, edge_features = self.edge_conv2(node_features, edge_index, torch.cat((edge_weights, edge_weights), dim=0),
                                           edge_features)

        edge_features = nn.functional.leaky_relu(edge_features)

        # h, c = self.lstm(torch.cat((edge_features.squeeze(), edge_features_1d, edge_weights.unsqueeze(-1)), dim=-1), h)  # h is (hidden state, cell state)

        #  might be better to not let policy gradient backprob through fe extraction

        q = self.out_q(torch.cat((edge_features.squeeze(), edge_features_1d, edge_weights.unsqueeze(-1)), dim=-1))

        side_loss = (side_loss1 + side_loss2) / 2
        # p = nn.functional.softmax(q, -1)  # this alternatively
        return q, side_loss


class WrappedGcnEdgeAngle1dQ(torch.nn.Module):
    def __init__(self, *args):
        super(WrappedGcnEdgeAngle1dQ, self).__init__()
        self.module = GcnEdgeAngle1dQ(*args)

    def forward(self, state, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None):
        return self.module(state, sp_indices, edge_index, angles, edge_features_1d)
