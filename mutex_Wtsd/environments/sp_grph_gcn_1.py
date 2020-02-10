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
    def __init__(self, edge_ids, edge_features, gt_edge_weights, node_features, seg, gt_seg, affinities, action_aggression=0.1, penalize_diff_thresh=2000):
        super(SpGcnEnv, self).__init__()
        self.initial_edge_weights = edge_features
        self.node_features = node_features
        self.init_sp_seg = seg.squeeze().numpy()
        self.gt_seg = gt_seg.squeeze().numpy()
        self.affinities = affinities.squeeze().numpy()
        self.edge_ids = edge_ids
        self.gt_edge_weights = gt_edge_weights
        self.action_agression = action_aggression
        self.penalize_diff_thresh = penalize_diff_thresh
        self.stop_quality = 100 * len(edge_features)
        self.reset()

    def execute_action(self, actions):
        mask = (actions == 2).float().unsqueeze(-1) * (self.state + self.action_agression)
        mask += (actions == 1).float().unsqueeze(-1) * (self.state - self.action_agression)
        mask += (actions == 0).float().unsqueeze(-1) * self.state
        self.state = mask
        self.state = torch.min(self.state, torch.ones_like(self.state))
        self.state = torch.max(self.state, torch.zeros_like(self.state))

        reward = -np.abs(self.state - self.gt_edge_weights).squeeze()

        # calculate reward
        self.data_changed = torch.sum(torch.abs(self.state - self.initial_edge_weights))
        penalize_change = 0
        if self.data_changed > self.penalize_diff_thresh:
            penalize_change = (self.penalize_diff_thresh - self.data_changed) / np.prod(self.state.size()) * 10
        reward += (penalize_change * (actions != 0).float())

        total_reward = torch.sum(reward).item()
        self.counter += 1

        # check if finished
        if total_reward >= -5:
            reward += 5
            self.done = True
            self.iteration += 1
            self.last_reward = -inf

        self.acc_reward += total_reward
        return self.state, reward

    def reset_data(self, edge_ids, edge_features, gt_edge_weights, node_features, seg, gt_seg, affinities):
        self.initial_edge_weights = edge_features
        self.node_features = node_features
        self.init_sp_seg = seg.squeeze().numpy()
        self.gt_seg = gt_seg.squeeze().numpy()
        self.affinities = affinities.squeeze().numpy()
        self.edge_ids = edge_ids
        self.gt_edge_weights = gt_edge_weights
        self.reset()

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
        self.state = self.initial_edge_weights.clone()

