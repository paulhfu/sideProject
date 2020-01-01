from environments.environment_bc import Environment
from mutex_watershed import compute_partial_mws_prim_segmentation, compute_mws_prim_segmentation
from affogato.segmentation.mws import get_valid_edges
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import inf
from environments.mutex_wtsd_bc import MtxWtsdEnvBc

class MtxWtsdEnvDN(MtxWtsdEnvBc):
    def __init__(self, affs, separating_channel, offsets, strides, gt_affinities, separating_action, stop_cnt=15,
                 win_reward=7, use_bbox=False, writer=None, action_aggression=2, penalize_diff_thresh=15):
        super(MtxWtsdEnvDN, self).__init__(affs, separating_channel, offsets, strides, gt_affinities=gt_affinities,
                                           stop_cnt=stop_cnt, win_reward=win_reward)

        self.writer = writer

        self.use_bbox = use_bbox
        self.action_aggression = action_aggression
        self.penalize_diff_thresh = penalize_diff_thresh
        self.separating_action = separating_action

        self.sorted_mtxs = [[w, id] for id, w in zip(range(len(self.current_affs[:separating_channel].ravel())),
                                                     self.current_affs[:separating_channel].ravel())]
        self.sorted_attr = [[w, id] for id, w in zip(range(len(self.current_affs[:separating_channel].ravel())),
                                                     self.current_affs[separating_channel:].ravel())]
        self.sorted_attr.sort()
        self.sorted_mtxs.sort()

        self.bbox = np.array(self.img_shape)
        self._update_state()

    def _update_state(self):
        node_labeling, cut_edges, used_mtxs, neighbors_features = self._calc_prtl_wtsd()

        cut_edge_imgs = np.zeros(node_labeling.size*len(self.mtx_offsets), dtype=np.float)

        for edge_id in cut_edges + used_mtxs:
            cut_edge_imgs[edge_id] = 1
        cut_edge_imgs = cut_edge_imgs.reshape((len(self.mtx_offsets),)+self.img_shape)
        if self.use_bbox:
            ymax_vals, xmax_vals = self._bbox(cut_edge_imgs)
            self.bbox = [np.max(ymax_vals), np.max(xmax_vals)]

        if not any(self.bbox==0):
            self.state = np.concatenate((cut_edge_imgs[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]],
                                     self.current_affs[:, 0:self.bbox[0], 0:self.bbox[1]]),
                                    axis=0).astype(np.float)

        self.mtx_wtsd_max_iter += self.mtx_wtsd_iter_step

        if self.counter % 20 == 0:
            node_labeling = node_labeling.reshape(self.img_shape)
            if self.writer is not None:
                self.writer.add_image('res/igmRes', cm.prism(node_labeling / node_labeling.max()), self.counter)

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

    def execute_action(self, action):
        last_affs = self.current_affs.copy()
        if action[0] < self.separating_action:
            sorted_edges = self.sorted_attr
            edge_id = sorted_edges[action[0]][1]
            c = edge_id // (self.current_affs.shape[2] * self.current_affs.shape[1])
        else:
            sorted_edges = self.sorted_mtxs
            edge_id = sorted_edges[action[0] - self.separating_action][1]
            c = edge_id // (self.current_affs.shape[2] * self.current_affs.shape[1]) + self.mtx_separating_channel

        iv = edge_id % (self.current_affs.shape[2]*self.current_affs.shape[1])
        y = iv // self.current_affs.shape[2]
        x = iv % self.current_affs.shape[2]
        if action[1] == 0:
            self.current_affs[c, y, x] /= self.action_aggression
            sorted_edges[action[0]][0] /= self.action_aggression
            i = action[0]
            while i > 0 and sorted_edges[i][0] < sorted_edges[i-1][0]:
                sorted_edges[i], sorted_edges[i-1] = sorted_edges[i-1], sorted_edges[i]
                i -= 1
        if action[1] == 1:
            self.current_affs[c, y, x] = max(1, self.current_affs[c, y, x] * self.action_aggression)
            sorted_edges[action[0]][0] = max(1, sorted_edges[action[0]][0] * self.action_aggression)
            i = action[0]
            while i < len(sorted_edges)-1 and sorted_edges[i][0] > sorted_edges[i+1][0]:
                sorted_edges[i], sorted_edges[i+1] = sorted_edges[i+1], sorted_edges[i]
                i += 1

        self._update_state()
        data_changed = np.sum(np.abs(self.current_affs - self.initial_affs))
        penalize_change = 0
        if data_changed > self.penalize_diff_thresh:
            penalize_change = self.penalize_diff_thresh - data_changed

        reward = - 10 + penalize_change + \
                 np.sum(self.state[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]].ravel() * 2 - 1 ==
                        self.attr_gt_affs[:, 0:self.bbox[0], 0:self.bbox[1]].ravel()) - \
                 np.sum(self.state[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]].ravel() !=
                        self.attr_gt_affs[:, 0:self.bbox[0], 0:self.bbox[1]].ravel())

        if reward < self.last_reward:
            keep_old_state = True
            self.current_affs = last_affs.copy()
        else:
            keep_old_state = False
            self.last_reward = reward

        if self.writer is not None:
            self.writer.add_scalar("step/reward", reward, self.counter + (self.stop_cnt * self.iteration))

        self.counter += 1

        # check if finished
        if reward + 10 >= self.stop_quality:
            reward = 100
            self.done = True
            self.iteration += 1

        if self.counter > self.stop_cnt:
            reward = -100
            self.done = True
            self.iteration += 1

        return self.state, reward, keep_old_state

    def reset(self):
        self.done = False
        self.acc_reward = 0
        self.last_reward = -inf
        self.counter = 0
        self.current_affs = self.initial_affs.copy()
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self._update_state()


