from mutex_watershed import compute_mws_segmentation_cstm
from matplotlib import cm
import numpy as np
import utils
import torch
from utils import pca_svd, ind_spat_2_flat, ind_flat_2_spat
from math import inf
import matplotlib.pyplot as plt
from environments.environment_bc import Environment


class SpGcnEnv(Environment):
    def __init__(self, edge_ids, edge_features, diff_to_gt, gt_edge_weights, node_features, seg, gt_seg, affinities, action_aggression=0.1, writer=None, angles=None):
        super(SpGcnEnv, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.init_sp_seg = seg.squeeze().numpy()
        self.gt_seg = gt_seg.squeeze().numpy()
        self.affinities = affinities.squeeze().numpy()
        self.edge_ids = edge_ids
        self.gt_edge_weights = gt_edge_weights
        self.action_agression = action_aggression
        self.penalize_diff_thresh = diff_to_gt * 1.2
        self.stop_quality = 0

        self.reset()
        self.writer = writer
        self.writer_idx1 = 0
        self.writer_idx2 = 0
        self.writer_idx3 = 0
        self.edge_angles = angles

    def execute_action(self, actions, episode=None):
        last_diff = (self.state - self.gt_edge_weights).squeeze().abs()
        mask = (actions == 2).float().unsqueeze(-1) * (self.state + self.action_agression)
        mask += (actions == 1).float().unsqueeze(-1) * (self.state - self.action_agression)
        mask += (actions == 0).float().unsqueeze(-1) * self.state
        self.state = mask

        diff = last_diff - (self.state - self.gt_edge_weights).squeeze().abs()
        reward = (diff > 0).float()*1 - (diff < 0).float()*2
        reward -= (((self.state - self.gt_edge_weights).squeeze().abs() > 0.1) & (actions == 0)).float() * 2

        self.state = torch.min(self.state, torch.ones_like(self.state))
        self.state = torch.max(self.state, torch.zeros_like(self.state))

        # calculate reward
        self.data_changed = torch.sum(torch.abs(self.state - self.initial_edge_weights))
        penalize_change = 0
        if self.data_changed > self.penalize_diff_thresh or self.counter > 90:
            # penalize_change = (self.penalize_diff_thresh - self.data_changed) / np.prod(self.state.size()) * 10
            self.done = True
            reward -= 5
            self.iteration += 1
        reward += (penalize_change * (actions != 0).float())

        total_reward = torch.sum(reward).item()
        self.counter += 1

        # check if finished
        quality = (self.state - self.gt_edge_weights).squeeze().abs().sum()
        if quality < self.stop_quality:
            reward += 5
            print('##################success######################')
            self.done = True
            self.iteration += 1
            if self.writer is not None:
                self.writer.add_text("winevent", str(episode) + ', ' + str(self.counter) + ', qual: ' + str(quality))
                self.writer_idx1 += 1
        if self.writer is not None:
            self.writer.add_scalar("step/quality", quality, self.writer_idx2)
            self.writer_idx2 += 1

        self.acc_reward = total_reward
        return self.state, reward

    def update_data(self, edge_ids, edge_features, diff_to_gt, gt_edge_weights, node_features, seg, gt_seg, affinities, angles=None):
        self.edge_features = edge_features
        self.node_features = node_features
        self.penalize_diff_thresh = diff_to_gt * 1.5
        self.init_sp_seg = seg.squeeze().numpy()
        self.gt_seg = gt_seg.squeeze().numpy()
        self.affinities = affinities.squeeze().numpy()
        self.edge_ids = edge_ids
        self.gt_edge_weights = gt_edge_weights
        self.edge_angles = angles

    def show_current_soln(self):
        affs = np.expand_dims(self.affinities, axis=1)
        boundary_input = np.mean(affs, axis=0)
        mc_seg1 = utils.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                             self.initial_edge_weights.squeeze().cpu().numpy(), boundary_input)
        mc_seg = utils.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                            self.state.squeeze().cpu().numpy(), boundary_input)
        gt_mc_seg = utils.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                               self.gt_edge_weights.squeeze().cpu().numpy(), boundary_input)
        mc_seg = cm.prism(mc_seg / mc_seg.max())
        mc_seg1 = cm.prism(mc_seg1 / mc_seg1.max())
        seg = cm.prism(self.init_sp_seg / self.init_sp_seg.max())
        gt_mc_seg = cm.prism(gt_mc_seg / gt_mc_seg.max())
        plt.imshow(np.concatenate((mc_seg1, mc_seg, gt_mc_seg, seg)));
        plt.show()

    def get_current_soln(self):
        affs = np.expand_dims(self.affinities, axis=1)
        boundary_input = np.mean(affs, axis=0)
        mc_seg = utils.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                            self.state.squeeze().cpu().numpy(), boundary_input)
        return mc_seg

    def reset(self):
        self.done = False
        self.iteration = 0
        self.acc_reward = 0
        self.last_reward = -inf
        self.counter = 0
        self.state = self.edge_features[:, 0].clone()
