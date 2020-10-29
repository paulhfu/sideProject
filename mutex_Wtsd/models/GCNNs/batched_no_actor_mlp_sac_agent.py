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

        self.actor = PolicyNet(self.args.n_embedding_features+1, 2, device, writer)
        self.critic = DoubleQValueNet(self.args.n_embedding_features+1, 1, self.args, device, writer)
        self.critic_tgt = DoubleQValueNet(self.args.n_embedding_features+1, 1, self.args, device, writer)

    def forward(self, raw, gt_edges=None, sp_indices=None, edge_index=None, angles=None, round_n=None,
                sub_graphs=None, sep_subgraphs=None, actions=None, post_input=False, policy_opt=False):

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
                out, _ = self.actor(node_features, edge_index, angles, gt_edges, post_input)
                mu, log_std = out.chunk(2, dim=-1)
                mu, log_std = mu.squeeze(), log_std.squeeze()

                if post_input and self.writer is not None:
                    self.writer.add_histogram("hist_logits/loc", mu.view(-1).detach().cpu().numpy(), self.writer_counter)
                    self.writer.add_histogram("hist_logits/scale", log_std.view(-1).detach().cpu().numpy(), self.writer_counter)
                    self.writer_counter += 1

                # constrain log_std inside [log_std_min, log_std_max]
                log_std = torch.tanh(log_std)
                log_std_min, log_std_max = self.log_std_bounds
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

                std = log_std.exp()

                # dist = TruncNorm(mu, std, 0, 1, 0.005)
                dist = SigmNorm(mu, std, self.args.s_subgraph, sub_graphs)
                actions = dist.rsample()

            q1, q2, _ = self.critic_tgt(node_features, actions, edge_index, angles, sub_graphs, gt_edges, post_input)
            # q1, q2 = q1[sub_graphs].view(-1, self.args.s_subgraph), q1[sub_graphs].view(-1, self.args.s_subgraph)
            q1, q2 = q1.squeeze(), q1.squeeze()
            return dist, q1, q2, actions

        q1, q2, _ = self.critic(node_features, actions, edge_index, angles, sub_graphs, gt_edges, post_input)
        # q1, q2 = q1[sub_graphs].view(-1, self.args.s_subgraph), q1[sub_graphs].view(-1, self.args.s_subgraph)
        q1, q2 = q1.squeeze(), q1.squeeze()
        return q1, q2


class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, device, writer):
        super(PolicyNet, self).__init__()
        self.gcn = Gcnn(n_in_features, n_classes, device, writer)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input):

        actor_stats, gcn_side_loss = self.gcn(node_features, edge_index, angles, gt_edges, post_input)
        return actor_stats, gcn_side_loss

class DoubleQValueNet_no_mlp(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, device, writer=None):
        super(DoubleQValueNet_no_mlp, self).__init__()
        self.fe1 = QGcnn(n_in_features, n_classes, device, writer)
        self.fe2 = QGcnn(n_in_features, n_classes, device, writer)

    def forward(self, node_features, edge_index, angles, gt_edges, actions, post_input):
        actions = actions.unsqueeze(-1)
        val1, gcn_side_loss1 = self.fe1(node_features, edge_index, angles, gt_edges, actions, post_input)
        val2, gcn_side_loss2 = self.fe1(node_features, edge_index, angles, gt_edges, actions, post_input)
        return val1, val2, gcn_side_loss2 + gcn_side_loss1

class DoubleQValueNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, args, device, writer):
        super(DoubleQValueNet, self).__init__()

        self.args = args

        self.gcn1 = QGcnn(n_in_features, n_in_features, device, writer)
        self.norm_features = torch.nn.BatchNorm1d(n_in_features, track_running_stats=False)

        self.value1 = nn.Sequential(
            nn.Linear(self.args.s_subgraph * n_in_features, 256),
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

        self.gcn2 = QGcnn(n_in_features, n_in_features, device, writer)
        self.value2 = nn.Sequential(
            nn.Linear(self.args.s_subgraph * n_in_features, 256),
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
        actions = actions.unsqueeze(-1)
        sg_edge_features1, gcn_side_loss1 = self.gcn1(node_features, edge_index, angles, gt_edges, actions, post_input)
        sg_edge_features1 = self.norm_features(sg_edge_features1)

        sg_edge_features1 = sg_edge_features1[sub_graphs].view(-1, self.args.s_subgraph * sg_edge_features1.shape[1])

        sg_edge_features2, gcn_side_loss2 = self.gcn2(node_features, edge_index, angles, gt_edges, actions, post_input)
        sg_edge_features2 = self.norm_features(sg_edge_features2)

        sg_edge_features2 = sg_edge_features2[sub_graphs].view(-1, self.args.s_subgraph * sg_edge_features2.shape[1])

        return self.value1(sg_edge_features1), self.value2(sg_edge_features2), gcn_side_loss1 + gcn_side_loss2

