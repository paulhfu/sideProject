from matplotlib import cm
import numpy as np
import torch
import elf
import nifty
from utils import general
from math import inf
import matplotlib.pyplot as plt
from environments.environment_bc import Environment
from utils.reward_functions import FocalReward, FullySupervisedReward, ObjectLevelReward, UnSupervisedReward, \
    GraphDiceReward, GlobalSparseReward, SubGraphDiceReward, HoughCircles, HoughCircles_lg, HoughCirclesOnSp
from utils.graphs import separate_nodes, collate_edges, get_edge_indices, get_angles_in_rag
from rag_utils import find_dense_subgraphs

class SpGcnEnv(Environment):
    def __init__(self, args, device, writer=None, writer_counter=None, win_event_counter=None):
        super(SpGcnEnv, self).__init__()

        self.reset()
        self.args = args
        self.device = device
        self.writer = writer
        self.writer_counter = writer_counter
        self.discrete_action_space = False
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)

        if self.args.reward_function == 'fully_supervised':
            self.reward_function = FullySupervisedReward(env=self)
        elif self.args.reward_function == 'sub_graph_dice':
            self.reward_function = SubGraphDiceReward(env=self)
        elif self.args.reward_function == 'defining_rules_edge_based':
            self.reward_function = HoughCircles(env=self, range_num=[8, 10],
                                                range_rad=[max(self.args.data_shape) // 18,
                                                           max(self.args.data_shape) // 15], min_hough_confidence=0.7)
        elif self.args.reward_function == 'defining_rules_sp_based':
            self.reward_function = HoughCirclesOnSp(env=self, range_num=[8, 10],
                                                range_rad=[max(self.args.data_shape) // 18,
                                                           max(self.args.data_shape) // 15], min_hough_confidence=0.7)
        elif self.args.reward_function == 'defining_rules_lg':
            assert False
        else:
            self.reward_function = UnSupervisedReward(env=self)

    def execute_action(self, actions, logg_vals=None, post_stats=False):
        last_diff = (self.sg_current_edge_weights - self.sg_gt_edge_weights).squeeze().abs()

        self.b_current_edge_weights = actions.clone()
        self.sg_current_edge_weights = actions[self.b_subgraph_indices].view(-1, self.args.s_subgraph)

        self.current_soln = self.get_current_soln(self.b_current_edge_weights)
        reward = self.reward_function.get(actions, self.current_soln)
        # reward = self.reward_function.get(actions, self.get_current_soln(self.b_gt_edge_weights))
        # reward = self.reward_function.get(actions=self.sg_current_edge_weights)

        quality = (self.sg_current_edge_weights - self.sg_gt_edge_weights).squeeze().abs().sum().item()
        self.counter += 1
        if self.counter >= self.args.max_episode_length:
            self.done = True

        total_reward = torch.sum(reward).item()

        if self.writer is not None and post_stats:
            self.writer.add_scalar("step/quality", quality, self.writer_counter.value())
            self.writer.add_scalar("step/avg_return", reward.mean(), self.writer_counter.value())
            if self.writer_counter.value() % 10 == 0:
                self.writer.add_histogram("step/pred_mean", self.sg_current_edge_weights.view(-1).cpu().numpy(), self.writer_counter.value() // 10)
                current_soln = self.get_current_soln(self.b_current_edge_weights)
                gt_soln = self.get_current_soln(self.b_gt_edge_weights)
                fig, (a1, a2, a3, a4) = plt.subplots(1, 4, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                a1.imshow(self.raw[0].cpu().permute(1,2,0).squeeze(), cmap='hot')
                a1.set_title('raw image')
                a2.imshow(cm.prism(self.init_sp_seg[0].cpu() / self.init_sp_seg[0].max().item()))
                a2.set_title('superpixels')
                a3.imshow(cm.prism(gt_soln[0].cpu()/gt_soln[0].max().item()))
                a3.set_title('gt')
                a4.imshow(cm.prism(current_soln[0].cpu()/current_soln[0].max().item()))
                a4.set_title('prediction')
                self.writer.add_figure("image/state", fig, self.writer_counter.value() // 10)
            self.writer.add_scalar("step/gt_mean", self.sg_gt_edge_weights.mean().item(), self.writer_counter.value())
            self.writer.add_scalar("step/gt_std", self.sg_gt_edge_weights.std().item(), self.writer_counter.value())
            if logg_vals is not None:
                for key, val in logg_vals.items():
                    self.writer.add_scalar("step/" + key, val, self.writer_counter.value())
            self.writer_counter.increment()

        self.acc_reward = total_reward
        return self.get_state(), reward, quality

    def get_state(self):
        return torch.cat([self.raw, self.init_sp_seg_edge], 1), self.init_sp_seg, self.b_edge_ids, self.sp_indices, \
               self.b_edge_angles, self.b_subgraph_indices, self.sep_subgraphs, self.counter, self.b_gt_edge_weights, self.e_offs

    def update_data(self, b_edge_ids, edge_features, diff_to_gt, gt_edge_weights, sp_seg, raw, gt):
        self.gt_seg = gt.squeeze(1)
        self.raw = raw
        self.init_sp_seg = sp_seg.squeeze(1)
        self.b_edge_angles = torch.cat([get_angles_in_rag(edges, sp) for edges, sp in zip(b_edge_ids, self.init_sp_seg)]).unsqueeze(-1)
        self.init_sp_seg_edge = torch.cat([(-self.max_p(-sp_seg) != sp_seg).float(), (self.max_p(sp_seg) != sp_seg).float()], 1)

        self.n_nodes = [edge_ids.max() + 1 for edge_ids in b_edge_ids]
        b_subgraphs, sep_subgraphs = find_dense_subgraphs([edge_ids.transpose(0, 1).cpu().numpy() for edge_ids in b_edge_ids], self.args.s_subgraph)
        self.sep_subgraphs = torch.from_numpy(sep_subgraphs.astype(np.int64))
        self.b_edge_ids, (self.n_offs, self.e_offs) = collate_edges(b_edge_ids)
        b_subgraphs = [torch.from_numpy(sg.astype(np.int64)).to(self.device).view(-1, 2).transpose(0, 1) + self.n_offs[i] for i, sg in enumerate(b_subgraphs)]
        self.sg_offs = [0]
        for i in range(len(b_subgraphs)):
            self.sg_offs.append(self.sg_offs[-1] + b_subgraphs[i].shape[-1])
        self.b_subgraphs = torch.cat(b_subgraphs, 1)

        self.b_subgraph_indices = get_edge_indices(self.b_edge_ids, self.b_subgraphs)
        self.b_gt_edge_weights = torch.cat(gt_edge_weights, 0)
        self.sg_gt_edge_weights = self.b_gt_edge_weights[self.b_subgraph_indices].view(-1, self.args.s_subgraph)
        self.sg_current_edge_weights = torch.ones_like(self.sg_gt_edge_weights) / 2

        self.b_initial_edge_weights = torch.cat([edge_fe[:, 0] for edge_fe in edge_features], dim=0)
        self.b_current_edge_weights = self.b_initial_edge_weights.clone()
        self.current_soln = self.get_current_soln(self.b_current_edge_weights)

        stacked_superpixels = [[sp_seg[i] == n for n in range(n_node)] for i, n_node in enumerate(self.n_nodes)]
        self.sp_indices = [[torch.nonzero(sp, as_tuple=False).cpu() for sp in stacked_superpixel] for stacked_superpixel in stacked_superpixels]

        # cs = self.get_current_soln(self.b_gt_edge_weights)
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(cm.prism(self.gt_seg[0].detach().cpu().numpy() / self.gt_seg[0].detach().cpu().numpy().max()));
        # ax1.set_title('gt')
        # ax2.imshow(cm.prism(self.init_sp_seg[0].detach().cpu().numpy() / self.init_sp_seg[0].detach().cpu().numpy().max()));
        # ax2.set_title('sp')
        # ax3.imshow(cm.prism(cs[0].detach().cpu().numpy() / cs[0].detach().cpu().numpy().max()));
        # ax3.set_title('mc')
        # plt.show()
        # a=1

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

    def get_current_soln(self, b_edge_weights):
        p_min = 0.001
        p_max = 1.
        segmentations = []
        for i in range(1, len(self.e_offs)):
            probs = b_edge_weights[self.e_offs[i-1]:self.e_offs[i]]
            # probs = self.b_gt_edge_weights[self.e_offs[i-1]:self.e_offs[i]]
            edges = self.b_edge_ids[:, self.e_offs[i-1]:self.e_offs[i]] - self.n_offs[i-1]
            costs = (p_max - p_min) * probs + p_min
            # probabilities to costs
            costs = (torch.log((1. - costs) / costs)).detach().cpu().numpy()
            graph = nifty.graph.undirectedGraph(self.n_nodes[i-1])
            graph.insertEdges(edges.T.cpu().numpy())

            node_labels = elf.segmentation.multicut.multicut_kernighan_lin(graph, costs)

            mc_seg = torch.zeros_like(self.init_sp_seg[i-1])
            for j, lbl in enumerate(node_labels):
                mc_seg += (self.init_sp_seg[i-1] == j).float() * lbl

            segmentations.append(mc_seg)
        return torch.stack(segmentations, dim=0)
        # return torch.rand_like(self.init_sp_seg)

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


