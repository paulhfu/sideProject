from environments.environment_bc import Environment
from mutex_watershed import compute_partial_mws_prim_segmentation, compute_mws_prim_segmentation
from affogato.segmentation.mws import get_valid_edges
import matplotlib.pyplot as plt
from mutex_watershed import compute_mws_segmentation_cstm
from matplotlib import cm
from utils import ind_spat_2_flat, ind_flat_2_spat
import numpy as np
from math import inf
from environments.mutex_wtsd_bc import MtxWtsdEnvBc

class MtxWtsdEnvMNM(MtxWtsdEnvBc):
    def __init__(self, affinities, separating_channel, offsets, strides, gt_affinities, stop_cnt=80,
                 win_reward=-10, lost_bound=-10, writer=None, only_prop_improv=False):
        sc = separating_channel + len(offsets[separating_channel:])
        affinities = np.concatenate((affinities[:separating_channel], affinities[separating_channel:], affinities[separating_channel:]))
        offsets = offsets[:separating_channel] + 2 * offsets[separating_channel:]
        super(MtxWtsdEnvMNM, self).__init__(affinities, sc, offsets, strides, gt_affinities=gt_affinities,
                                            stop_cnt=stop_cnt, win_reward=win_reward, lost_bound=lost_bound,
                                            separating_channel_lr_attr=separating_channel)
        self.writer = writer
        self.only_prop_improv = only_prop_improv
        self.node_labeling, neighbors, self.cutting_edges, self.mutexes = self._calc_wtsd()
        self.neighbors = np.concatenate((np.expand_dims(ind_flat_2_spat(neighbors[:, 0], self.node_labeling.shape), 1),
                                         np.expand_dims(ind_flat_2_spat(neighbors[:, 1], self.node_labeling.shape), 1))
                                        , axis=1)
        self.gt_seg, _, _, _ = self._calc_gt_wtsd()
        self.n_neighbors = len(self.neighbors)
        # neighb = np.zeros((self.n_neighbors, 2))
        # for i, n in enumerate(self.neighbors):
        #     neighb[i] = [self.node_labeling[n[0,0], n[0,1]], self.node_labeling[n[1,0], n[1,1]]]
        self.ttl_cnt = 0
        # self.rewards = self.calculate_reward(self.neighbors, self.gt_seg, self.node_labeling)
        self._update_state()
        self.terminal_state = np.zeros_like(self.state)
        self.win_reward = 1
        self.penalty_reward = -1

    def update_data(self, affinities=None, gt_affinities=None):
        if affinities is not None:
            affinities = np.concatenate((affinities[:self.separating_channel_lr_attr],
                                         affinities[self.separating_channel_lr_attr:],
                                         affinities[self.separating_channel_lr_attr:]))
            super(MtxWtsdEnvMNM, self).update_data(affinities, gt_affinities)
        self.node_labeling, neighbors, self.cutting_edges, self.mutexes = self._calc_wtsd()
        self.neighbors = np.concatenate((np.expand_dims(ind_flat_2_spat(neighbors[:, 0], self.node_labeling.shape), 1),
                                         np.expand_dims(ind_flat_2_spat(neighbors[:, 1], self.node_labeling.shape), 1))
                                        , axis=1)
        self.gt_seg, _, _, _ = self._calc_gt_wtsd()
        self.n_neighbors = len(self.neighbors)
        # self.rewards = self.calculate_reward(self.neighbors, self.gt_seg, self.node_labeling)
        # assert len(self.rewards) == self.n_neighbors
        assert len(self.mutexes) == self.n_neighbors
        assert len(self.cutting_edges) == self.n_neighbors

    def _update_state(self):
        neigh = self.neighbors[self.counter]
        mask = (self.node_labeling == self.node_labeling[neigh[0, 0], neigh[0, 1]]) + \
               (self.node_labeling == self.node_labeling[neigh[1, 0], neigh[1, 1]])

        self.state = np.stack([mask, self.node_labeling], axis=0).astype(np.float)

        self.mtx_wtsd_max_iter += self.mtx_wtsd_iter_step

        if self.ttl_cnt % 20 == 0:
            if self.writer is not None:
                self.writer.add_image('res/igmRes', cm.prism(self.node_labeling / self.node_labeling.max()), self.counter)

    def _bbox(self, array2d_c):
        assert len(array2d_c.shape) == 3
        ymax_vals = []
        xmax_vals = []
        for array2d in array2d_c:
            y = np.where(np.any(array2d, axis=1))
            x = np.where(np.any(array2d, axis=0))
            ymin, ymax = y[0][[0, -1]] if len(y[0]) != 0 else (0, 0)
            xmin, xmax = x[0][[0, -1]] if len(x[0]) != 0 else (0, 0)
            ymax_vals.append(ymax)
            xmax_vals.append(xmax)
        return ymax_vals, xmax_vals

    def calculate_reward(self, neighbors, gt_seg, new_seg):
        rewards = np.zeros([len(neighbors)] + [2])
        self.masks = []
        # if self.n_neighbors == 36:
        #    self.show_current_soln()
        # if self.n_neighbors == 33:
        #     a = np.zeros((2, 40, 40))
        #     a = a.ravel()
        #     self.show_current_soln()
        #     for edge in self.cutting_edges:
        #         for e in edge:
        #             a[e] = 1
        #     a = a.reshape((2, 40, 40))
        #     for img in a:
        #         plt.imshow(img);
        #         plt.show();

        for idx, neighbor in enumerate(neighbors):
            mask_n1, mask_n2 = new_seg == new_seg[neighbor[0, 0], neighbor[0, 1]], new_seg == new_seg[neighbor[1, 0], neighbor[1, 1]]
            mask = mask_n1 + mask_n2
            obj_area = np.sum(mask)
            mskd_gt_seg = mask * gt_seg
            mskd_new_seg = mask * new_seg
            n_obj_gt = np.unique(mskd_gt_seg)
            n_obj_new = np.unique(mskd_new_seg)
            n_obj_gt = n_obj_gt[1:] if n_obj_gt[0] == 0 else n_obj_gt
            if len(n_obj_gt) == 1:
                rewards[idx] = [self.win_reward, self.penalty_reward]
            else:
                n_obj_new = n_obj_new[1:] if n_obj_new[0] == 0 else n_obj_new
                n_obj_pnlty = - abs(len(n_obj_new) - len(n_obj_gt)) * 10
                assert len(n_obj_new) == 2
                overlaps = np.zeros([len(n_obj_gt)] + [2])
                for j, obj in enumerate(n_obj_gt):
                    mask_gt = mskd_gt_seg == obj
                    overlaps[j] = np.sum(mask_gt * mask_n1) / np.sum(mask_n1), \
                                  np.sum(mask_gt * mask_n2) / np.sum(mask_n2)
                    # plt.imshow(mask_gt * mask_n1);plt.show();
                    # plt.imshow(mask_gt * mask_n2);plt.show();
                if np.sum(overlaps.max(axis=1) > 0.5) >= 2:
                    rewards[idx] = [self.penalty_reward, self.win_reward]
                else:
                    rewards[idx] = [self.win_reward, self.penalty_reward]
                    # if self.n_neighbors == 36:
                    #     plt.imshow(mskd_gt_seg);
                    #     plt.show();
                    #     plt.imshow(mskd_new_seg);
                    #     plt.show();
            self.masks.append((np.concatenate([cm.prism(mask_n1 / mask_n1.max()), cm.prism(mask_n2 / mask_n2.max()), cm.prism(mask / mask.max())]), mask))
            # img1 = np.concatenate([np.concatenate([cm.prism(new_seg / new_seg.max()), cm.prism(mask / mask.max())], axis=1),
            #                       np.concatenate([cm.prism(mskd_gt_seg / mskd_gt_seg.max()), cm.prism(mskd_new_seg / mskd_new_seg.max())], axis=1)], axis=0)
            # import matplotlib.pyplot as plt;plt.imshow(img1);plt.show();
            # a=1
        return rewards

    def execute_action(self, action, learn=True):
        last_valid_edges = self.valid_edges.copy()
        reward = self.calculate_reward([self.neighbors[self.counter]], self.gt_seg, self.node_labeling)[0, action]
        self.acc_reward += reward
        if action == 0:
            # self.show_current_soln(mask=self.masks[self.counter][1])
            # self.show_current_soln()
            valid_edges = self.valid_edges.ravel()
            valid_edges[self.mutexes[self.counter]] = False
            valid_edges[(self.mutexes[self.counter] - self.n_mtxs)] = True
            self.valid_edges = valid_edges.reshape(self.valid_edges.shape)
            self.counter = 0
            self.update_data()
        else:
            self.counter += 1
        self.ttl_cnt += 1

        if self.counter >= self.n_neighbors or (reward == self.penalty_reward and learn):
            # if reward == -1:
            #     reward += 500
            # reward /= self.counter
            self.done = True
            self.iteration += 1
            return self.terminal_state, reward, False
        self._update_state()

        keep_old_state = False
        if reward == -10 and self.only_prop_improv:
            keep_old_state = True
            self.valid_edges = last_valid_edges
        # if action == 0:
        #     # self.show_current_soln(mask=self.masks[self.counter-1][1])
        #     # self.show_current_soln(mask=self.get_gt_soln()-1)
        #     # self.show_current_soln()
        #     self.counter = 0
        #     self.update_data()

        return self.state, reward, keep_old_state

    def execute_opt_policy(self):
        self.show_current_soln()
        # self.show_current_soln(mask=self.get_gt_soln()-1)
        while not self.done:
            action = reward = self.calculate_reward([self.neighbors[self.counter]], self.gt_seg,
                                                    self.node_labeling).argmax(axis=1)
            # neigh = np.where(actions == 0)[0][0]
            # self.counter = neigh
            # self.ttl_cnt = 0
            _, reward, _ = self.execute_action(action)
            # actions = self.calculate_reward(self.neighbors, self.gt_seg, self.node_labeling).argmax(axis=1)
            print("reward: ", reward)
            print("neighbors: ", self.n_neighbors)
        self.show_current_soln()
        print("steps: ", self.ttl_cnt)
        self.show_current_soln(mask=self.get_gt_soln()-1)
        a=1

    def reset(self):
        self.done = False
        self.acc_reward = 0
        self.counter = 0
        self.ttl_cnt = 0
        self.current_affs = self.initial_affs.copy()
        self.valid_edges = self.initial_valid_edges.copy()
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self._update_state()
        self.update_data()
        self.terminal_state = np.zeros_like(self.state)

    def execute_all_actions(self, actions):
        assert len(actions) == len(self.neighbors)
        for idx, action in enumerate(actions):
            if action == 0:  # merge
                valid_edges = self.valid_edges.ravel()
                valid_edges[self.mutexes[idx]] = False
                valid_edges[self.mutexes[idx] - self.n_mtxs] = True
                self.valid_edges = valid_edges.reshape(self.valid_edges.shape)
