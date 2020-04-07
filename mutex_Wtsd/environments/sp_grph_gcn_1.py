from matplotlib import cm
import numpy as np
import torch
from utils import general
from math import inf
import matplotlib.pyplot as plt
from environments.environment_bc import Environment
from utils.reward_functions import FullySupervisedReward, ObjectLevelReward, UnSupervisedReward


class SpGcnEnv(Environment):
    def __init__(self, args, device, writer=None, writer_counter=None, win_event_counter=None):
        super(SpGcnEnv, self).__init__()
        self.stop_quality = 0

        self.reset()
        self.args = args
        self.device = device
        self.writer = writer
        self.writer_counter = writer_counter
        self.win_event_counter = win_event_counter

        if self.args.reward_function == 'fully_supervised':
            self.reward_function = FullySupervisedReward(env=self)
        elif self.args.reward_function == 'object_level_reward':
            self.reward_function = ObjectLevelReward(env=self)
        else:
            self.reward_function = UnSupervisedReward(env=self)

    def execute_action(self, actions, episode=None):
        last_diff = (self.state[0] - self.gt_edge_weights).squeeze().abs()
        mask = (actions == 2).float() * (self.state[0] + self.args.action_agression)
        mask += (actions == 1).float() * (self.state[0] - self.args.action_agression)
        mask += (actions == 0).float() * self.state[0]
        self.state[0] = mask

        self.state[0] = self.state[0].clamp(min=0, max=1)
        reward = self.reward_function.get(last_diff, actions, self.get_current_soln()).to(self.device)

        self.data_changed = torch.sum(torch.abs(self.state[0] - self.edge_features[:, 0]))
        penalize_change = 0
        if self.data_changed > self.penalize_diff_thresh or self.counter > self.args.max_episode_length:
            # penalize_change = (self.penalize_diff_thresh - self.data_changed) / np.prod(self.state.size()) * 10
            self.done = True
            reward -= 5
            self.iteration += 1
        reward += (penalize_change * (actions != 0).float())

        total_reward = torch.sum(reward).item()
        self.counter += 1

        # check if finished
        quality = (self.state[0] - self.gt_edge_weights).squeeze().abs().sum()
        if quality < self.stop_quality:
            reward += 5
            self.done = True
            self.iteration += 1
            self.win_event_counter.increment()
        if self.writer is not None and self.done:
            self.writer.add_scalar("step/quality", quality, self.writer_counter.value())
            self.writer.add_scalar("step/stop_quality", self.stop_quality, self.writer_counter.value())
            self.writer.add_scalar("step/n_wins", self.win_event_counter.value(), self.writer_counter.value())
            self.writer.add_scalar("step/steps_needed", self.counter, self.writer_counter.value())
            self.writer.add_scalar("step/win_loose_ratio", (self.win_event_counter.value()+1) /
                                   (self.writer_counter.value()+1), self.writer_counter.value())
            self.writer_counter.increment()

        self.acc_reward = total_reward
        self.state[1] = self.get_current_soln()
        return [self.state[0].clone(), self.state[1].clone()], reward

    def update_data(self, edge_ids, edge_features, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles,
                    affinities, gt):
        self.gt_seg = gt
        self.affinities = affinities
        self.initial_edge_weights = edge_features[:, 0]
        self.edge_features = edge_features
        self.stacked_superpixels = [node_labeling == n for n in nodes]
        self.sp_indices = [sp.nonzero() for sp in self.stacked_superpixels]
        self.raw = raw
        self.penalize_diff_thresh = diff_to_gt * 4
        self.init_sp_seg = node_labeling.squeeze()
        self.edge_ids = edge_ids
        self.gt_edge_weights = gt_edge_weights
        self.edge_angles = angles
        self.state = [self.edge_features[:, 0].clone(), None]
        self.state = [self.edge_features[:, 0], self.get_current_soln()]

    def show_current_soln(self):
        affs = np.expand_dims(self.affinities, axis=1)
        boundary_input = np.mean(affs, axis=0)
        mc_seg1 = general.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                               self.initial_edge_weights.squeeze().cpu().numpy(), boundary_input)
        mc_seg = general.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                              self.state.squeeze().cpu().numpy(), boundary_input)
        gt_mc_seg = general.multicut_from_probas(self.init_sp_seg, self.edge_ids.cpu().t().contiguous().numpy(),
                                                 self.gt_edge_weights.squeeze().cpu().numpy(), boundary_input)
        mc_seg = cm.prism(mc_seg / mc_seg.max())
        mc_seg1 = cm.prism(mc_seg1 / mc_seg1.max())
        seg = cm.prism(self.init_sp_seg / self.init_sp_seg.max())
        gt_mc_seg = cm.prism(gt_mc_seg / gt_mc_seg.max())
        # plt.imshow(np.concatenate((np.concatenate((mc_seg1, mc_seg), 0), np.concatenate((gt_mc_seg, seg), 0)), 1));
        plt.imshow(cm.prism(self.init_sp_seg/self.init_sp_seg.max()))
        plt.show()
        plt.imshow(gt_mc_seg)
        plt.show()
        a=1

    def get_current_soln(self):
        affs = np.expand_dims(self.affinities, axis=1)
        boundary_input = np.mean(affs, axis=0)
        mc_seg = general.multicut_from_probas(self.init_sp_seg.squeeze().cpu(), self.edge_ids.cpu().t().contiguous().numpy(),
                                              self.state[0].squeeze().cpu().numpy(), boundary_input)
        return torch.from_numpy(mc_seg.astype(np.float))

    def reset(self):
        self.done = False
        self.iteration = 0
        self.acc_reward = 0
        self.last_reward = -inf
        self.counter = 0


