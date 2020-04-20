import torch
from models.GCNNs.cstm_message_passing import NodeConv1, EdgeConv1
from models.GCNNs.cstm_message_passing import GcnEdgeConv, EdgeConv, NodeConv, EdgeConvNoEdge, SpatEdgeConv, \
    SpatNodeConv
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d
from models.sp_embed_unet import SpVecsUnet
from utils.truncated_normal import TruncNorm
from torch_geometric.nn import GCNConv, GATConv


class GcnEdgeAngle1dPQA_dueling(torch.nn.Module):
    def __init__(self, n_raw_channels, n_embedding_channels, n_edge_features_in, n_edge_classes, exp_steps, p_sigma,
                 device, density_eval_range):
        super(GcnEdgeAngle1dPQA_dueling, self).__init__()
        self.fe_ext = SpVecsUnet(n_raw_channels, n_embedding_channels, device)
        self.p_sigma = p_sigma
        self.density_eval_range = density_eval_range
        self.exp_steps = exp_steps
        self.node_conv1 = NodeConv1(n_embedding_channels, n_embedding_channels)
        self.edge_conv1 = EdgeConv1(n_embedding_channels, n_embedding_channels)
        self.node_conv2 = NodeConv1(n_embedding_channels, n_embedding_channels)
        self.edge_conv2 = EdgeConv1(n_embedding_channels, n_embedding_channels, use_init_edge_feats=True,
                                    n_channels_in=n_embedding_channels)

        # self.lstm = nn.LSTMCell(n_embedding_channels + n_edge_features_in + 1, hidden_size)

        self.out_p1 = nn.Linear(n_embedding_channels + n_edge_features_in, 256)
        self.out_p2 = nn.Linear(256, n_edge_classes)
        self.out_q1 = nn.Linear(n_embedding_channels + n_edge_features_in + 1, 256)
        self.out_q2 = nn.Linear(256, n_edge_classes)
        self.device = device

    def forward(self, state, action_behav, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None,
                stats_only=False):
        edge_weights = state[0].to(self.device)
        input = torch.stack((state[1], state[2])).unsqueeze(0).to(self.device)
        if sp_indices is None:
            return self.fe_ext(input)
        if edge_features_1d is None:
            return self.fe_ext(input, sp_indices)
        node_features = self.fe_ext(input, sp_indices)
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

        p = self.out_p2(self.out_p1(torch.cat((edge_features.squeeze(), edge_features_1d), dim=-1)))
        # want this between 0 and one therefore the normalozation, since we expect at least one edge to be 0 and at least one to be 1 in the gt
        p = torch.sigmoid(p)

        p_dis = TruncNorm(loc=p.squeeze().detach(), scale=self.p_sigma, a=0, b=1, eval_range=self.density_eval_range)
        if stats_only:
            return p.squeeze(), p_dis

        v = self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, action_behav.unsqueeze(-1)), dim=-1))
        v = self.out_q2(v)

        a = self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, action_behav.unsqueeze(-1)), dim=-1))
        a = self.out_q2(a)

        action = p_dis.sample().unsqueeze(-1)
        exp_adv = self.out_q2(self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, action), dim=-1)))

        for i in range(self.exp_steps):
            action = p_dis.sample().unsqueeze(-1)
            exp_adv = exp_adv + self.out_q2(
                self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, action), dim=-1)))

        exp_adv = exp_adv / (self.exp_steps + 1)

        q = v + a - exp_adv

        with torch.set_grad_enabled(False):
            sampled_action = p_dis.sample().unsqueeze(-1)
            v_prime = self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, sampled_action), dim=-1))
            v_prime = self.out_q2(v_prime)

            a_prime = self.out_q1(torch.cat((edge_features.squeeze(), edge_features_1d, sampled_action), dim=-1))
            a_prime = self.out_q2(a_prime)
            q_prime = v_prime + a_prime - exp_adv

        return p.squeeze(), q.squeeze(), v.squeeze(), p_dis, sampled_action.squeeze(), q_prime.squeeze()


class WrappedGcnEdgeAngle1dPQA_dueling(torch.nn.Module):
    def __init__(self, *args):
        super(WrappedGcnEdgeAngle1dPQA_dueling, self).__init__()
        self.module = GcnEdgeAngle1dPQA_dueling(*args)

    def forward(self, state, action_behav, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None, stats_only=False):
        return self.module(state, action_behav, sp_indices, edge_index, angles, edge_features_1d, stats_only)
