import torch
import torch.nn as nn
from models.sp_embed_unet import SpVecsUnet, SpVecsUnet
from utils.sigmoid_normal1 import SigmNorm


class GcnEdgeAC(torch.nn.Module):
    def __init__(self, cfg, n_raw_channels, n_embedding_channels, n_edge_classes, device, softmax=True,
                 writer=None):
        super(GcnEdgeAC, self).__init__()
        self.writer = writer
        self.log_std_bounds = cfg.diag_gaussian_actor.log_std_bounds
        self.device = device
        self.writer_counter = 0

        self.fe_ext = SpVecsUnet(n_embedding_channels, n_raw_channels, n_embedding_channels, device, writer)

        self.actor = nn.Sequential(
            nn.Linear(n_embedding_channels + 1, 256),
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
            nn.Linear(256, 2 * n_edge_classes),
        )

        self.norm_features = torch.nn.BatchNorm1d(n_embedding_channels, track_running_stats=False)

        self.critic = DoubleValueNet(n_embedding_channels + 2, n_edge_classes)
        self.critic_tgt = DoubleValueNet(n_embedding_channels + 2, n_edge_classes)

    def forward(self, raw=None, edge_features=None, gt_edges=None, sp_indices=None, edge_index=None, angles=None, round_n=None,
                actions=None, post_input=False, policy_opt=False):

        if edge_features is None:
            if sp_indices is None:
                return self.fe_ext(raw)
            edge_features, embeddings, _ = self.fe_ext(raw, edge_index, angles, gt_edges, sp_indices, post_input)
            edge_features = self.norm_features(edge_features)
            edge_features = torch.cat(
                [edge_features, torch.ones([edge_features.shape[0], 1], device=edge_features.device) * round_n], -1)

        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                mu, log_std = self.actor(edge_features).chunk(2, dim=-1)
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

            q1, q2 = self.critic_tgt(torch.cat((edge_features, actions.unsqueeze(-1)), dim=-1))
            q1, q2 = q1.squeeze(), q2.squeeze()
            return dist, q1, q2, actions

        q1, q2 = self.critic(torch.cat((edge_features, actions.unsqueeze(-1)), dim=-1))
        q1, q2 = q1.squeeze(), q2.squeeze()
        return q1, q2


class DoubleValueNet(torch.nn.Module):
    def __init__(self, n_in_features, n_out_features):
        super(DoubleValueNet, self).__init__()
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
            nn.Linear(256, n_out_features),
        )

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
            nn.Linear(256, n_out_features),
        )

    def forward(self, input):
        return self.value1(input), self.value2(input)


class WrappedGcnEdgeAC(torch.nn.Module):
    def __init__(self, *args):
        super(WrappedGcnEdgeAC, self).__init__()
        self.module = GcnEdgeAC(*args)

    def forward(self, state, sp_indices=None, edge_index=None, angles=None, edge_features_1d=None):
        return self.module(state, sp_indices, edge_index, angles, edge_features_1d)