from matplotlib import cm
import numpy as np
import torch
from utils import general
from math import inf
import matplotlib.pyplot as plt
from environments.environment_bc import Environment
from utils.reward_functions import FocalReward, FullySupervisedReward, ObjectLevelReward, UnSupervisedReward, \
    GraphDiceReward, GlobalSparseReward


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
        self.discrete_action_space = False

        if self.args.reward_function == 'fully_supervised':
            self.reward_function = FullySupervisedReward(env=self)
        elif self.args.reward_function == 'object_level':
            self.reward_function = ObjectLevelReward(env=self)
        elif self.args.reward_function == 'graph_dice':
            self.reward_function = GraphDiceReward(env=self)
        elif self.args.reward_function == 'focal':
            self.reward_function = FocalReward(env=self)
        elif self.args.reward_function == 'global_sparse':
            self.reward_function = GlobalSparseReward(env=self)
        else:
            self.reward_function = UnSupervisedReward(env=self)

    def execute_action(self, actions, logg_vals=None):
        last_diff = (self.current_edge_weights - self.gt_edge_weights).squeeze().abs()

        self.current_edge_weights = actions.clone()

        reward = self.reward_function.get(last_diff, actions, self.get_current_soln()).to(self.device)

        quality = (self.current_edge_weights - self.gt_edge_weights).squeeze().abs().sum().item()
        self.counter += 1
        if self.counter >= self.args.max_episode_length:
            if quality < self.stop_quality:
                # reward += 2
                self.win = True
            else:
                a=1
                # reward -= 1

            self.done = True
            self.win_event_counter.increment()

        total_reward = torch.sum(reward).item()

        if self.writer is not None and self.done:
            self.writer.add_scalar("step/quality", quality, self.writer_counter.value())
            self.writer.add_scalar("step/stop_quality", self.stop_quality, self.writer_counter.value())
            self.writer.add_scalar("step/n_wins", self.win_event_counter.value(), self.writer_counter.value())
            self.writer.add_scalar("step/steps_needed", self.counter, self.writer_counter.value())
            self.writer.add_scalar("step/win_loose_ratio", (self.win_event_counter.value()+1) /
                                   (self.writer_counter.value()+1), self.writer_counter.value())
            self.writer.add_scalar("step/pred_mean", self.current_edge_weights.mean(), self.writer_counter.value())
            self.writer.add_scalar("step/pred_std", self.current_edge_weights.std(), self.writer_counter.value())
            self.writer.add_scalar("step/gt_mean", self.gt_edge_weights.mean(), self.writer_counter.value())
            self.writer.add_scalar("step/gt_std", self.gt_edge_weights.std(), self.writer_counter.value())
            if logg_vals is not None:
                for key, val in logg_vals.items():
                    self.writer.add_scalar("step/" + key, val, self.writer_counter.value())
            self.writer_counter.increment()

        self.acc_reward = total_reward
        state_pixels = torch.stack([self.raw, self.init_sp_seg, self.get_current_soln()], dim=0)
        return (state_pixels, self.edge_ids, self.sp_indices, self.edge_angles, self.counter), reward, quality

    def get_state(self):
        state_pixels = torch.stack([self.raw, self.init_sp_seg, self.get_current_soln()], dim=0)
        return state_pixels, self.edge_ids, self.sp_indices, self.edge_angles, self.counter

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
        self.current_edge_weights = torch.ones_like(gt_edge_weights) / 2

    def show_current_soln(self):
        affs = np.expand_dims(self.affinities, axis=1)
        boundary_input = np.mean(affs, axis=0)
        mc_seg1 = general.multicut_from_probas(self.init_sp_seg.cpu(), self.edge_ids.cpu().t().contiguous().numpy(),
                                               self.initial_edge_weights.squeeze().cpu().numpy(), boundary_input)
        mc_seg = general.multicut_from_probas(self.init_sp_seg.cpu(), self.edge_ids.cpu().t().contiguous().numpy(),
                                              self.current_edge_weights.squeeze().cpu().numpy(), boundary_input)
        gt_mc_seg = general.multicut_from_probas(self.init_sp_seg.cpu(), self.edge_ids.cpu().t().contiguous().numpy(),
                                                 self.gt_edge_weights.squeeze().cpu().numpy(), boundary_input)
        mc_seg = cm.prism(mc_seg / mc_seg.max())
        mc_seg1 = cm.prism(mc_seg1 / mc_seg1.max())
        seg = cm.prism(self.init_sp_seg.cpu() / self.init_sp_seg.cpu().max())
        gt_mc_seg = cm.prism(gt_mc_seg / gt_mc_seg.max())
        plt.imshow(np.concatenate((np.concatenate((mc_seg1, mc_seg), 0), np.concatenate((gt_mc_seg, seg), 0)), 1));
        plt.show()
        a=1
        ####################
        # init mc # gt seg #
        ####################
        # curr mc # sp seg #
        ####################

    def get_current_soln(self):
        # affs = np.expand_dims(self.affinities, axis=1)
        # boundary_input = np.mean(affs, axis=0)
        # mc_seg = general.multicut_from_probas(self.init_sp_seg.squeeze().cpu(), self.edge_ids.cpu().t().contiguous().numpy(),
        #                                       self.current_edge_weights.squeeze().cpu().numpy(), boundary_input)
        # return torch.from_numpy(mc_seg.astype(np.float32))
        return torch.rand_like(self.init_sp_seg.squeeze())

    def get_rag_and_edge_feats(self, reward, edges):
        edge_indices = []
        seg = self.init_sp_seg.clone()
        for edge in self.edge_ids.t():
            n1, n2 = self.sp_indices[edge[0]], self.sp_indices[edge[1]]
            dis = torch.cdist(n1.float(), n2.float())
            dis = (dis <= 1).nonzero()
            inds_n1 = n1[dis[:, 0].unique()]
            inds_n2 = n2[dis[:, 1].unique()]
            edge_indices.append(torch.cat((inds_n1, inds_n2), 0))
        for indices in edge_indices:
            seg[indices[:, 0], indices[:, 1]] = 600
        seg = cm.prism(seg.cpu().numpy() / seg.cpu().numpy().max())
        plt.imshow(seg)
        plt.show()

    def reset(self):
        self.done = False
        self.win = False
        self.acc_reward = 0
        self.last_reward = -inf
        self.counter = 0


