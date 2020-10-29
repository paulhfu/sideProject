import torch
from models.GCNNs.cstm_message_passing import NodeConv, EdgeConv
import torch.nn.functional as F
import torch.nn as nn
from models.sp_embed_unet import SpVecsUnet
import matplotlib.pyplot as plt
from utils.general import _pca_project, plt_bar_plot
from utils.truncated_normal import TruncNorm
from models.GCNNs.gcnn import Gcnn, QGcnn
from utils.sigmoid_normal1 import SigmNorm
# import gpushift


class GcnEdgeAC(torch.nn.Module):
    def __init__(self, cfg, args, device, writer=None):
        super(GcnEdgeAC, self).__init__()
        self.writer = writer
        self.args = args
        self.cfg = cfg
        self.log_std_bounds = cfg.diag_gaussian_actor.log_std_bounds
        self.device = device
        self.writer_counter = 0
        n_q_vals = args.s_subgraph
        if "sg_rew" in args.algorithm:
            n_q_vals = 1

        self.fe_ext = SpVecsUnet(self.args.n_raw_channels, self.args.n_embedding_features, device, writer)

        self.actor = PolicyNet(self.args.n_embedding_features * self.args.s_subgraph, self.args.s_subgraph * 2, args, device, writer)
        self.critic = DoubleQValueNet((1 + self.args.n_embedding_features) * self.args.s_subgraph, n_q_vals, args, device, writer)
        self.critic_tgt = DoubleQValueNet((1 + self.args.n_embedding_features) * self.args.s_subgraph, n_q_vals, args, device, writer)

    def forward(self, raw, gt_edges=None, sp_indices=None, edge_index=None, angles=None, round_n=None,
                sub_graphs=None, actions=None, post_input=False, policy_opt=False):

        if sp_indices is None:
            return self.fe_ext(raw)
        embeddings = self.fe_ext(raw)
        node_features = []
        for i, sp_ind in enumerate(sp_indices):
            post_inp = False
            if post_input and i == 0:
                post_inp = True
            node_features.append(self.fe_ext.get_node_features(raw[i].squeeze(), embeddings[i].squeeze(), sp_ind, post_input=post_inp))

        # create one large unconnected graph where each connected component corresponds to one image
        node_features = torch.cat(node_features, dim=0)
        node_features = torch.cat(
            [node_features, torch.ones([node_features.shape[0], 1], device=node_features.device) * round_n], -1)

        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge

        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out = self.actor(node_features, edge_index, angles, sub_graphs, gt_edges, post_input)
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

            q1, q2 = self.critic_tgt(node_features, actions, edge_index, angles, sub_graphs, gt_edges, post_input)
            return dist, q1, q2, actions

        q1, q2 = self.critic(node_features, actions, edge_index, angles, sub_graphs, gt_edges, post_input)
        return q1, q2


class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, args, device, writer):
        super(PolicyNet, self).__init__()
        self.args = args

        self.gcn = Gcnn(self.args.n_embedding_features + 1, self.args.n_embedding_features, device, writer)
        self.norm_features = torch.nn.BatchNorm1d(self.args.n_embedding_features, track_running_stats=False)

        self.stats = nn.Sequential(
            nn.Linear(n_in_features, 256),
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

    def forward(self, node_features, edge_index, angles, sub_graphs, gt_edges, post_input):

        edge_features, gcn_side_loss = self.gcn(node_features, edge_index, angles, gt_edges, post_input)
        edge_features = self.norm_features(edge_features)

        sg_edge_features = edge_features[sub_graphs].view(-1, self.args.s_subgraph * edge_features.shape[1])
        return self.stats(sg_edge_features)


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, args, device, writer):
        super(DoubleQValueNet, self).__init__()

        self.args = args
        self.gcn1 = Gcnn(self.args.n_embedding_features + 1, self.args.n_embedding_features, device, writer)
        self.norm_features = torch.nn.BatchNorm1d(self.args.n_embedding_features, track_running_stats=False)
        self.value1 = nn.Sequential(
            nn.Linear(n_in_features, 256),
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

        self.gcn2 = Gcnn(self.args.n_embedding_features + 1, self.args.n_embedding_features, device, writer)
        self.value2 = nn.Sequential(
            nn.Linear(n_in_features, 256),
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

    def forward(self, node_features, actions, edge_index, angles, sub_graphs, gt_edges, post_input):

        edge_features, gcn_side_loss = self.gcn1(node_features, edge_index, angles, gt_edges, post_input)
        edge_features = self.norm_features(edge_features)

        sg_edge_features1 = edge_features[sub_graphs].view(-1, self.args.s_subgraph * edge_features.shape[1])
        sg_edge_features1 = torch.cat((sg_edge_features1, actions), dim=-1)

        edge_features, gcn_side_loss = self.gcn2(node_features, edge_index, angles, gt_edges, post_input)
        edge_features = self.norm_features(edge_features)

        sg_edge_features2 = edge_features[sub_graphs].view(-1, self.args.s_subgraph * edge_features.shape[1])
        sg_edge_features2 = torch.cat((sg_edge_features2, actions), dim=-1)

        return self.value1(sg_edge_features1), self.value2(sg_edge_features2)

