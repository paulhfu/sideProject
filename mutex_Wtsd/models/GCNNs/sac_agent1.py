import torch
from models.GCNNs.cstm_message_passing import NodeConv1, EdgeConv1
import torch.nn.functional as F
import torch.nn as nn
from models.sp_embed_unet import SpVecsUnet, SpVecsUnetGcn
import matplotlib.pyplot as plt
from utils.general import _pca_project, plt_bar_plot
from utils.truncated_normal import TruncNorm
from models.GCNNs.gcnn import Gcnn, QGcnn
from utils.sigmoid_normal1 import SigmNorm
# import gpushift


class GcnEdgeAC_1(torch.nn.Module):
    def __init__(self, cfg, n_raw_channels, n_embedding_channels, n_edge_classes, device, softmax=True,
                 writer=None):
        super(GcnEdgeAC_1, self).__init__()
        self.writer = writer
        self.log_std_bounds = cfg.diag_gaussian_actor.log_std_bounds
        self.device = device
        self.writer_counter = 0

        self.fe_ext = SpVecsUnet(n_raw_channels, n_embedding_channels, device, writer)

        self.actor = PolicyNet(n_embedding_channels + 1, n_embedding_channels + 1, n_edge_classes * 2, device, writer)
        self.critic = DoubleQValueNet(n_embedding_channels + 1, n_embedding_channels + 1, n_edge_classes, device, writer)
        self.critic_tgt = DoubleQValueNet(n_embedding_channels + 1, n_embedding_channels + 1, n_edge_classes, device, writer)

    def forward(self, raw=None, edge_features=None, gt_edges=None, sp_indices=None, edge_index=None, angles=None, round_n=None,
                actions=None, post_input=False, policy_opt=False):

        if edge_features is None:
            if sp_indices is None:
                return self.fe_ext(raw)
            node_features, embeddings = self.fe_ext(raw, sp_indices, post_input)
            node_features = torch.cat(
                [node_features, torch.ones([node_features.shape[0], 1], device=node_features.device) * round_n], -1)

        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, _ = self.actor(node_features, edge_index, angles, gt_edges, post_input)
                mu, log_std = out.chunk(2, dim=-1)
                mu, log_std = mu.squeeze(), log_std.squeeze()

                if post_input and self.writer is not None:
                    self.writer.add_scalar("mean_logits/loc", mu.mean().item(), self.writer_counter)
                    self.writer.add_scalar("mean_logits/scale", log_std.mean().item(), self.writer_counter)
                    self.writer_counter += 1

                # constrain log_std inside [log_std_min, log_std_max]
                log_std = torch.tanh(log_std)
                log_std_min, log_std_max = self.log_std_bounds
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

                std = log_std.exp()

                # dist = TruncNorm(mu, std, 0, 1, 0.005)
                dist = SigmNorm(mu, std)
                actions = dist.rsample()

            q1, q2, _ = self.critic_tgt(node_features, edge_index, angles, gt_edges, actions.unsqueeze(-1), post_input)
            q1, q2 = q1.squeeze(), q2.squeeze()
            return dist, q1, q2, actions

        q1, q2, _ = self.critic(node_features, edge_index, angles, gt_edges, actions.unsqueeze(-1), post_input)
        q1, q2 = q1.squeeze(), q2.squeeze()
        return q1, q2


class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_out_features, n_classes, device, writer=None):
        super(PolicyNet, self).__init__()
        self.fe1 = Gcnn(n_in_features, n_out_features, device, writer)
        self.norm_features = torch.nn.BatchNorm1d(n_out_features, track_running_stats=False)
        self.value1 = nn.Sequential(
            nn.Linear(n_out_features, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, node_features, edge_index, angles, gt_edges, post_input=False):
        fe, gcn_side_loss = self.fe1(node_features, edge_index, angles, gt_edges, post_input)
        return self.value1(self.norm_features(fe)), gcn_side_loss


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, n_in_features, n_out_features, n_classes, device, writer=None):
        super(DoubleQValueNet, self).__init__()
        self.fe1 = QGcnn(n_in_features, n_out_features, device, writer)
        self.value1 = nn.Sequential(
            nn.Linear(n_out_features, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, n_classes),
        )
        self.norm_features = torch.nn.BatchNorm1d(n_out_features, track_running_stats=False)
        self.fe2 = QGcnn(n_in_features, n_out_features, device, writer)
        self.value2 = nn.Sequential(
            nn.Linear(n_out_features, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, node_features, edge_index, angles, gt_edges, actions, post_input=False):
        fe1, gcn_side_loss1 = self.fe1(node_features, edge_index, angles, gt_edges, actions, post_input)
        fe2, gcn_side_loss2 = self.fe1(node_features, edge_index, angles, gt_edges, actions, post_input)
        return self.value1(self.norm_features(fe1)), self.value2(self.norm_features(fe2)), gcn_side_loss2 + gcn_side_loss1

