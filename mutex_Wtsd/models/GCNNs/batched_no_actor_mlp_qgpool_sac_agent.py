import torch
import torch.nn.functional as F
import torch.nn as nn
# from models.sp_embed_with_affs import SpVecsUnet
from models.sp_embed_unet import SpVecsUnet
import matplotlib.pyplot as plt
from utils.general import _pca_project, plt_bar_plot
from utils.truncated_normal import TruncNorm
from models.GCNNs.gcnn import Gcnn, QGcnn, GlobalEdgeGcnn
from utils.sigmoid_normal1 import SigmNorm
import numpy as np
from losses.rag_contrastive_loss import RagContrastiveLoss


class GcnEdgeAC(torch.nn.Module):
    def __init__(self, cfg, device, writer=None):
        super(GcnEdgeAC, self).__init__()
        self.writer = writer
        self.cfg = cfg
        self.log_std_bounds = self.cfg.sac.diag_gaussian_actor.log_std_bounds
        self.device = device
        self.writer_counter = 0

        self.fe_ext = SpVecsUnet(self.cfg.fe.n_raw_channels, self.cfg.fe.n_embedding_features, device, writer)

        self.actor = PolicyNet(self.cfg.fe.n_embedding_features, 2, cfg.model.n_hidden, cfg.model.hl_factor, device, writer)
        self.critic = DoubleQValueNet(self.cfg.sac.s_subgraph, self.cfg.fe.n_embedding_features, 1,
                                      cfg.model.n_hidden, cfg.model.hl_factor, device, writer)
        self.critic_tgt = DoubleQValueNet(self.cfg.sac.s_subgraph, self.cfg.fe.n_embedding_features, 1,
                                          cfg.model.n_hidden, cfg.model.hl_factor, device, writer)

        self.log_alpha = torch.tensor([np.log(self.cfg.sac.init_temperature)] * len(self.cfg.sac.s_subgraph)).to(device)
        self.log_alpha.requires_grad = True

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
            return self.fe_ext(raw, post_input)
        with torch.set_grad_enabled(embeddings_opt):
            embeddings = self.fe_ext(raw, post_input)
        node_feats = []
        for i, sp_ind in enumerate(sp_indices):
            n_f = self.fe_ext.get_node_features(embeddings[i], sp_ind)
            node_feats.append(n_f)

        node_features = torch.cat(node_feats, dim=0)

        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge

        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, side_loss = self.actor(node_features, edge_index, angles, gt_edges, post_input)
                mu, log_std = out.chunk(2, dim=-1)
                mu, log_std = mu.squeeze(), log_std.squeeze()

                if post_input and self.writer is not None:
                    self.writer.add_histogram("hist_logits/loc", mu.view(-1).detach().cpu().numpy(), self.writer_counter)
                    self.writer.add_histogram("hist_logits/scale", log_std.view(-1).detach().cpu().numpy(), self.writer_counter)
                    self.writer_counter += 1

                log_std = torch.tanh(log_std)
                log_std_min, log_std_max = self.log_std_bounds
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

                std = log_std.exp()

                dist = SigmNorm(mu, std)
                actions = dist.rsample()

            q1, q2, sl = self.critic_tgt(node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input)
            side_loss = (side_loss + sl) / 2
            if policy_opt:
                return dist, q1, q2, actions, side_loss
            else:
                # this means either exploration,critic opt or embedding opt
                return dist, q1, q2, actions, embeddings, side_loss

        q1, q2, side_loss = self.critic(node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input)
        return q1, q2, side_loss


class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, n_hidden_layer, hl_factor, device, writer):
        super(PolicyNet, self).__init__()
        self.gcn = Gcnn(n_in_features, n_classes, n_hidden_layer, hl_factor, device, writer)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input):
        actor_stats, side_loss = self.gcn(node_features, edge_index, angles, gt_edges, post_input)
        return actor_stats, side_loss


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, s_subgraph, n_in_features, n_classes, n_hidden_layer, hl_factor, device, writer):
        super(DoubleQValueNet, self).__init__()

        self.s_subgraph = s_subgraph

        self.gcn1_1 = QGcnn(n_in_features, n_in_features, n_hidden_layer, hl_factor, device, writer)
        self.gcn2_1 = QGcnn(n_in_features, n_in_features, n_hidden_layer, hl_factor, device, writer)

        self.gcn1_2, self.gcn2_2 = [], []

        for i, ssg in enumerate(self.s_subgraph):
            self.gcn1_2.append(GlobalEdgeGcnn(n_in_features, n_in_features, ssg//2, hl_factor, device, writer))
            self.gcn2_2.append(GlobalEdgeGcnn(n_in_features, n_in_features, ssg//2, hl_factor, device, writer))
            super(DoubleQValueNet, self).add_module(f"gcn1_2_{i}", self.gcn1_2[-1])
            super(DoubleQValueNet, self).add_module(f"gcn2_2_{i}", self.gcn2_2[-1])

        self.value1 = nn.Sequential(
            nn.Linear(n_in_features, hl_factor * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, hl_factor * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, n_classes),
        )

        self.value2 = nn.Sequential(
            nn.Linear(n_in_features, hl_factor * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, hl_factor * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, n_classes),
        )

    def forward(self, node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input):
        actions = actions.unsqueeze(-1)
        _sg_edge_features1, side_loss = self.gcn1_1(node_features, edge_index, angles, gt_edges, actions, post_input)

        _sg_edge_features2, _side_loss = self.gcn2_1(node_features, edge_index, angles, gt_edges, actions, post_input)
        side_loss += _side_loss

        sg_edge_features1, sg_edge_features2 = [], []
        for i, sg_size in enumerate(self.s_subgraph):
            sub_sg_edge_features1 = _sg_edge_features1[sub_graphs[i]]
            sub_sg_edge_features2 = _sg_edge_features2[sub_graphs[i]]
            sg_edges = torch.cat([sep_subgraphs[i], torch.stack([sep_subgraphs[i][1], sep_subgraphs[i][0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge

            sub_sg_edge_features1, _side_loss = self.gcn1_2[i](sub_sg_edge_features1, sg_edges)
            side_loss += _side_loss
            sub_sg_edge_features2, _side_loss = self.gcn2_2[i](sub_sg_edge_features2, sg_edges)
            side_loss += _side_loss

            sg_edge_features1.append(self.value1(sub_sg_edge_features1.view(-1, sg_size,
                                                                        sub_sg_edge_features1.shape[-1]).mean(1)).squeeze())
            sg_edge_features2.append(self.value2(sub_sg_edge_features2.view(-1, sg_size,
                                                                        sub_sg_edge_features2.shape[-1]).mean(1)).squeeze())

        return sg_edge_features1, sg_edge_features2, side_loss / 4
