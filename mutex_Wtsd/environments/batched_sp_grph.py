from matplotlib import cm
import numpy as np
import torch
from utils import general
from math import inf
import matplotlib.pyplot as plt
from environments.environment_bc import Environment
from utils.reward_functions import FocalReward, FullySupervisedReward, ObjectLevelReward, UnSupervisedReward, \
    GraphDiceReward, GlobalSparseReward
from utils.graphs import separate_nodes, collate_edges, get_edge_indices
from rag_utils import find_dense_subgraphs

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
        last_diff = (self.b_current_edge_weights - self.b_gt_edge_weights).squeeze().abs()

        self.b_current_edge_weights = actions.clone()

        reward = self.reward_function.get(last_diff, actions, self.get_current_soln()).to(self.device)

        quality = (self.b_current_edge_weights - self.b_gt_edge_weights).squeeze().abs().sum().item()
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
            self.writer.add_scalar("step/pred_mean", self.b_current_edge_weights.mean(), self.writer_counter.value())
            self.writer.add_scalar("step/pred_std", self.b_current_edge_weights.std(), self.writer_counter.value())
            self.writer.add_scalar("step/gt_mean", self.b_gt_edge_weights.mean(), self.writer_counter.value())
            self.writer.add_scalar("step/gt_std", self.b_gt_edge_weights.std(), self.writer_counter.value())
            if logg_vals is not None:
                for key, val in logg_vals.items():
                    self.writer.add_scalar("step/" + key, val, self.writer_counter.value())
            self.writer_counter.increment()

        self.acc_reward = total_reward
        return self.get_state(), reward, quality

    def get_state(self):
        state_pixels = torch.cat([self.raw, self.init_sp_seg, self.get_current_soln()], dim=1)
        return state_pixels, self.b_edge_ids, self.sp_indices, self.b_edge_angles, self.counter, self.b_gt_edge_weights

    def update_data(self, b_edge_ids, edge_features, diff_to_gt, gt_edge_weights, node_labeling, raw, angles, gt):
        self.gt_seg = gt
        self.raw = raw
        self.init_sp_seg = node_labeling

        self.n_nodes = [edge_ids.max() + 1 for edge_ids in b_edge_ids]
        # b_subgraphs = find_dense_subgraphs([edge_ids.transpose(0, 1).cpu().numpy() for edge_ids in b_edge_ids], self.args.s_subgraph)
        self.b_edge_ids, (self.n_offs, self.e_offs) = collate_edges(b_edge_ids)
        # b_subgraphs = [torch.from_numpy(sg.astype(np.int64)).to(self.device).view(-1, 2).transpose(0, 1) + self.n_offs[i] for i, sg in enumerate(b_subgraphs)]
        # self.sg_offs = [0]
        # for i in range(len(b_subgraphs)):
        #     self.sg_offs.append(self.sg_offs[-1] + b_subgraphs[i].shape[0])
        # self.b_subgraphs = torch.cat(b_subgraphs, 1)
        #
        # self.b_subgraph_indices = get_edge_indices(self.b_edge_ids, self.b_subgraphs)
        self.b_gt_edge_weights = torch.cat(gt_edge_weights, 0)
        # self.sg_gt_edge_weights = self.b_gt_edge_weights[self.b_subgraph_indices].view(-1, self.args.s_subgraph)
        self.b_edge_angles = angles
        self.b_current_edge_weights = torch.ones_like(self.b_gt_edge_weights) / 2
        # self.sg_current_edge_weights = torch.ones_like(self.sg_gt_edge_weights) / 2

        self.b_initial_edge_weights = torch.cat([edge_fe[:, 0] for edge_fe in edge_features], dim=0)
        self.b_edge_features = torch.cat(edge_features, dim=0)

        stacked_superpixels = [[node_labeling[i] == n for n in range(n_node)] for i, n_node in enumerate(self.n_nodes)]
        self.sp_indices = [[sp.nonzero().cpu() for sp in stacked_superpixel] for stacked_superpixel in stacked_superpixels]

        # self.b_penalize_diff_thresh = diff_to_gt * 4
        # plt.imshow(self.get_current_soln_pic(1));plt.show()
        # return

    def get_batched_actions_from_global_graph(self, actions):
        b_actions = torch.zeros(size=(self.b_edge_ids.shape[1],))
        other = torch.zeros_like(self.b_subgraph_indices)
        for i in range(self.b_edge_ids.shape[1]):
            mask = (self.b_subgraph_indices == i)
            num = mask.float().sum()
            b_actions[i] = torch.where(mask, actions.float(), other.float()).sum() / num
        return b_actions

    def get_current_soln(self):
        # affs = np.expand_dims(self.affinities, axis=1)
        # boundary_input = np.mean(affs, axis=0)
        # mc_seg = general.multicut_from_probas(self.init_sp_seg.squeeze().cpu(), self.edge_ids.cpu().t().contiguous().numpy(),
        #                                       self.current_edge_weights.squeeze().cpu().numpy(), boundary_input)
        # return torch.from_numpy(mc_seg.astype(np.float32))
        return torch.rand_like(self.init_sp_seg)

    def get_current_soln_pic(self, b):
        b_actions = self.get_batched_actions_from_global_graph(self.sg_current_edge_weights.view(-1))
        b_gt = self.get_batched_actions_from_global_graph(self.sg_gt_edge_weights.view(-1))

        edge_ids = self.b_edge_ids[:, self.e_offs[b]: self.e_offs[b+1]] - self.n_offs[b]
        edge_ids = edge_ids.cpu().t().contiguous().numpy()
        boundary_input = self.b_initial_edge_weights[self.e_offs[b]: self.e_offs[b+1]].cpu().numpy()
        mc_seg1 = general.multicut_from_probas(self.init_sp_seg[b].squeeze().cpu(), edge_ids,
                                               self.b_initial_edge_weights[self.e_offs[b]: self.e_offs[b+1]].cpu().numpy(), boundary_input)
        mc_seg = general.multicut_from_probas(self.init_sp_seg[b].squeeze().cpu(), edge_ids,
                                              b_actions[self.e_offs[b]: self.e_offs[b+1]].cpu().numpy(), boundary_input)
        gt_mc_seg = general.multicut_from_probas(self.init_sp_seg[b].squeeze().cpu(), edge_ids,
                                                 b_gt[self.e_offs[b]: self.e_offs[b+1]].cpu().numpy(), boundary_input)
        mc_seg = cm.prism(mc_seg / mc_seg.max())
        mc_seg1 = cm.prism(mc_seg1 / mc_seg1.max())
        seg = cm.prism(self.init_sp_seg[b].squeeze().cpu() / self.init_sp_seg[b].cpu().max())
        gt_mc_seg = cm.prism(gt_mc_seg / gt_mc_seg.max())
        return np.concatenate((np.concatenate((mc_seg1, mc_seg), 0), np.concatenate((gt_mc_seg, seg), 0)), 1)
        ####################
        # init mc # gt seg #
        ####################
        # curr mc # sp seg #
        ####################

    def reset(self):
        self.done = False
        self.win = False
        self.acc_reward = 0
        self.last_reward = -inf
        self.counter = 0


