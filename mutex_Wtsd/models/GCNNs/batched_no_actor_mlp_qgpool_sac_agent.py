import torch
from models.GCNNs.cstm_message_passing import NodeConv1, EdgeConv1
import torch.nn.functional as F
import torch.nn as nn
# from models.sp_embed_with_affs import SpVecsUnet
from models.sp_embed_unet1 import SpVecsUnet
import matplotlib.pyplot as plt
from utils.general import _pca_project, plt_bar_plot
from utils.truncated_normal import TruncNorm
from models.GCNNs.gcnn import Gcnn, QGcnn, GlobalEdgeGcnn
from utils.sigmoid_normal1 import SigmNorm
import numpy as np
from losses.rag_contrastive_loss import ContrastiveLoss
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

        self.actor = PolicyNet(self.args.n_embedding_features, 2, device, writer)
        self.critic = DoubleQValueNet(self.args.n_embedding_features, 1, self.args, device, writer)
        self.critic_tgt = DoubleQValueNet(self.args.n_embedding_features, 1, self.args, device, writer)

        self.log_alpha = torch.tensor(np.log(self.cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.embedd_loss = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

    def forward(self, raw, sp_seg, gt_edges=None, sp_indices=None, edge_index=None, angles=None, round_n=None,
                sub_graphs=None, sep_subgraphs=None, actions=None, post_input=False, policy_opt=False, embeddings_opt=False):

        if sp_indices is None:
            return self.fe_ext(raw)
        with torch.set_grad_enabled(embeddings_opt):
            embeddings = self.fe_ext(raw)
        # self.embedd_loss.beta = self.alpha.item()
        # embedding_regularizer = self.embedd_loss(embeddings, sp_seg.long())
        node_feats = []
        # embedding_regularizer = 0
        for i, sp_ind in enumerate(sp_indices):
            post_inp = False
            if post_input and i == 0:
                post_inp = True
            n_f, _mass = self.fe_ext.get_node_features(raw[i].squeeze(), embeddings[i].detach().squeeze(), sp_ind, post_input=post_inp)
            # embedding_regularizer = embedding_regularizer + _reg
            node_feats.append(n_f)

        # embedding_regularizer = embedding_regularizer / len(sp_indices)

        # create one large unconnected graph where each connected component corresponds to one image
        node_features = torch.cat(node_feats, dim=0).detach()

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
                dist = SigmNorm(mu, std)
                actions = dist.rsample()

            q1, q2, _ = self.critic_tgt(node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input)
            # q1, q2 = q1[sub_graphs].view(-1, self.args.s_subgraph), q1[sub_graphs].view(-1, self.args.s_subgraph)
            q1, q2 = q1.squeeze(), q1.squeeze()
            if policy_opt:
                return dist, q1, q2, actions
            else:
                return dist, q1, q2, actions, embeddings


        q1, q2, _ = self.critic(node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input)
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


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, args, device, writer):
        super(DoubleQValueNet, self).__init__()

        self.args = args

        self.gcn1_1 = QGcnn(n_in_features, n_in_features, device, writer)
        self.gcn1_2 = GlobalEdgeGcnn(n_in_features, n_in_features, device, writer)
        self.l_relu = torch.nn.LeakyReLU()
        self.norm_features = torch.nn.BatchNorm1d(n_in_features, track_running_stats=False)

        self.gcn2_1 = QGcnn(n_in_features, n_in_features, device, writer)
        self.gcn2_2 = GlobalEdgeGcnn(n_in_features, n_in_features, device, writer)

        self.value1 = nn.Sequential(
            nn.Linear(n_in_features, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )

        self.value2 = nn.Sequential(
            nn.Linear(n_in_features, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )

    def forward(self, node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input):
        actions = actions.unsqueeze(-1)
        sg_edge_features1, _ = self.gcn1_1(node_features, edge_index, angles, gt_edges, actions, post_input)
        sg_edge_features1 = self.l_relu(sg_edge_features1)
        sg_edge_features1 = self.norm_features(sg_edge_features1)

        sg_edge_features2, _ = self.gcn2_1(node_features, edge_index, angles, gt_edges, actions, post_input)
        sg_edge_features2 = self.l_relu(sg_edge_features2)
        sg_edge_features2 = self.norm_features(sg_edge_features2)

        sg_edge_features1 = sg_edge_features1[sub_graphs]
        sg_edge_features2 = sg_edge_features2[sub_graphs]
        sg_edges = sep_subgraphs.view(-1, 2).T
        sg_edges = torch.cat([sg_edges, torch.stack([sg_edges[1], sg_edges[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge

        sg_edge_features1, _ = self.gcn1_2(sg_edge_features1, sg_edges, angles)
        sg_edge_features1 = self.l_relu(sg_edge_features1)
        sg_edge_features2, _ = self.gcn2_2(sg_edge_features2, sg_edges, angles)
        sg_edge_features2 = self.l_relu(sg_edge_features2)

        sg_edge_features1 = sg_edge_features1.view(-1, self.args.s_subgraph, sg_edge_features1.shape[-1]).mean(1)
        sg_edge_features2 = sg_edge_features2.view(-1, self.args.s_subgraph, sg_edge_features2.shape[-1]).mean(1)

        return self.value1(sg_edge_features1), self.value2(sg_edge_features2), None

