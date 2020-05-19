import torch
from models.GCNNs.cstm_message_passing import NodeConv1, EdgeConv1
import torch.nn.functional as F
import torch.nn as nn
from models.sp_embed_unet import SpVecsUnet


class GcnEdgeAngle1dQ(torch.nn.Module):
    def __init__(self, n_raw_channels, n_embedding_channels, n_edge_features_in, n_edge_classes, device, softmax=True,
                 writer=None):
        super(GcnEdgeAngle1dQ, self).__init__()
        self.writer = writer
        self.fe_ext = SpVecsUnet(n_raw_channels, n_embedding_channels, device)
        self.softmax = softmax
        n_embedding_channels += 1
        self.node_conv1 = NodeConv1(n_embedding_channels, n_embedding_channels, n_hidden_layer=5)
        self.edge_conv1 = EdgeConv1(n_embedding_channels, n_embedding_channels, n_embedding_channels, n_hidden_layer=5)
        self.node_conv2 = NodeConv1(n_embedding_channels, n_embedding_channels, n_hidden_layer=5)
        self.edge_conv2 = EdgeConv1(n_embedding_channels, n_embedding_channels, n_embedding_channels,
                                    use_init_edge_feats=True, n_init_edge_channels=n_embedding_channels,
                                    n_hidden_layer=5)

        # self.lstm = nn.LSTMCell(n_embedding_channels + n_edge_features_in + 1, hidden_size)

        self.out_q1 = nn.Linear(n_embedding_channels + n_edge_features_in + 1, 256)
        self.out_q2 = nn.Linear(256, n_edge_classes)
        self.device = device
        self.writer_counter = 0

    def forward(self, state, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None, round_n=None,
                post_input=False):
        edge_weights = state[0].to(self.device)
        input = state[2].unsqueeze(0).unsqueeze(0).to(self.device)

        if self.writer is not None and post_input:
            self.writer.add_image("image/state1", state[1].unsqueeze(0).cpu(), self.writer_counter)
            self.writer.add_image("image/state2", state[2].unsqueeze(0).cpu(), self.writer_counter)
            self.writer.add_image("image/state3", state[3].unsqueeze(0).cpu(), self.writer_counter)
            self.writer_counter += 1

        # input = torch.stack((state[1], state[2], state[3])).unsqueeze(0).to(self.device)
        if sp_indices is None:
            return self.fe_ext(input)
        if edge_features_1d is None:
            return self.fe_ext(input, sp_indices)
        node_features = self.fe_ext(input, sp_indices)
        node_features = torch.cat(
            [node_features, torch.ones([node_features.shape[0], 1], device=node_features.device) * round_n], -1)
        node_features, _ = self.node_conv1(node_features, edge_index, angles)
        node_features = nn.functional.leaky_relu(node_features)
        _, edge_features = self.edge_conv1(node_features, edge_index, torch.cat((edge_weights, edge_weights), dim=0))
        edge_features = nn.functional.leaky_relu(edge_features)
        node_features, _ = self.node_conv2(node_features, edge_index, angles)
        node_features = nn.functional.leaky_relu(node_features)
        _, edge_features = self.edge_conv2(node_features, edge_index, torch.cat((edge_weights, edge_weights), dim=0),
                                           edge_features)
        edge_features = nn.functional.leaky_relu(edge_features)

        # h, c = self.lstm(torch.cat((edge_features.squeeze(), edge_features_1d, edge_weights.unsqueeze(-1)), dim=-1), h)  # h is (hidden state, cell state)

        #  might be better to not let policy gradient backprob through fe extraction

        q = self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, edge_weights.unsqueeze(-1)), dim=-1))
        q = self.out_q2(q)

        # p = nn.functional.softmax(q, -1)  # this alternatively
        return q


class WrappedGcnEdgeAngle1dQ(torch.nn.Module):
    def __init__(self, *args):
        super(WrappedGcnEdgeAngle1dQ, self).__init__()
        self.module = GcnEdgeAngle1dQ(*args)

    def forward(self, state, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None):
        return self.module(state, sp_indices, edge_index, angles, edge_features_1d)
