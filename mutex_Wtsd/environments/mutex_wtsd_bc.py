from environments.environment_bc import Environment
from mutex_watershed import compute_mws_segmentation_cstm
from affogato.segmentation.mws import get_valid_edges
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import inf


class MtxWtsdEnvBc(Environment):
    def __init__(self, affinities, separating_channel, offsets, strides, win_reward, gt_affinities=None,
                 stop_cnt=15, separating_channel_lr_attr=None):
        super(MtxWtsdEnvBc, self).__init__(stop_cnt)

        self.separating_channel_lr_attr = separating_channel_lr_attr
        self.initial_affs = affinities
        self.current_affs = self.initial_affs.copy()
        self.mtx_separating_channel = separating_channel
        self.gt_separating_channel = separating_channel
        self.img_shape = affinities[0].shape
        self.n_nodes = np.prod(self.img_shape)
        self.n_mtxs = self.n_nodes * (len(offsets) - separating_channel)
        self.mtx_offsets = offsets
        self.gt_offsets = offsets
        self.mtx_strides = strides
        self.valid_edges = get_valid_edges((len(self.mtx_offsets),) + self.img_shape, self.mtx_offsets,
                                           self.mtx_separating_channel, self.mtx_strides, False)
        self.initial_valid_edges = self.valid_edges
        if self.separating_channel_lr_attr is not None:
            self.valid_edges[self.separating_channel_lr_attr:self.mtx_separating_channel] = False
            if gt_affinities is not None:
                self.gt_affs = gt_affinities
                self.gt_separating_channel = self.separating_channel_lr_attr
                self.gt_valid_edges = np.concatenate((self.valid_edges[:self.gt_separating_channel], self.valid_edges[self.mtx_separating_channel:]))
                self.gt_offsets = self.mtx_offsets[:self.gt_separating_channel] + self.mtx_offsets[self.mtx_separating_channel:]
                self.attr_gt_affs = (self.gt_affs[:self.gt_separating_channel] == 0).astype(np.float) \
                                    * self.gt_valid_edges[:separating_channel_lr_attr]
        if gt_affinities is not None:
            self.gt_affs = gt_affinities
            self.attr_gt_affs = (self.gt_affs[:self.mtx_separating_channel] == 0).astype(np.float) \
                                * self.valid_edges[:self.mtx_separating_channel]

        self.mtx_wtsd_start_iter = 1000000
        self.mtx_wtsd_iter_step = 50
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self.stop_quality = win_reward
        self.iteration = 0
        self.last_reward = -inf

    def update_data(self, affinities, gt_affinities=None):
        self.initial_affs = affinities
        self.current_affs = self.initial_affs.copy()
        self.valid_edges = get_valid_edges((len(self.mtx_offsets),) + self.img_shape, self.mtx_offsets,
                                           self.mtx_separating_channel, self.mtx_strides, False)
        self.initial_valid_edges = self.valid_edges
        if self.separating_channel_lr_attr is not None:
            self.valid_edges[self.separating_channel_lr_attr:self.mtx_separating_channel] = False
            if gt_affinities is not None:
                self.gt_affs = gt_affinities
                self.gt_separating_channel = self.separating_channel_lr_attr
                self.gt_valid_edges = np.concatenate((self.valid_edges[:self.gt_separating_channel], self.valid_edges[self.mtx_separating_channel:]))
                self.gt_offsets = self.mtx_offsets[:self.gt_separating_channel] + self.mtx_offsets[self.mtx_separating_channel:]
                self.attr_gt_affs = (self.gt_affs[:self.gt_separating_channel] == 0).astype(np.float) \
                                    * self.gt_valid_edges[:self.separating_channel_lr_attr]
        if gt_affinities is not None:
            self.gt_affs = gt_affinities
            self.attr_gt_affs = (self.gt_affs[:self.mtx_separating_channel] == 0).astype(np.float) \
                                * self.valid_edges[:self.mtx_separating_channel]

    def show_current_soln(self, mask=None):
        node_labeling = self.get_current_soln()
        if mask is not None:
            node_labeling = node_labeling*mask
        seg = cm.prism(node_labeling / node_labeling.max())
        plt.imshow(seg)
        plt.show()

    def show_gt_seg(self, mask=None):
        node_labeling = self.get_gt_soln()
        if mask is not None:
            node_labeling = node_labeling*mask
        seg = cm.prism(node_labeling / node_labeling.max())
        plt.imshow(seg)
        plt.show()

    def get_current_soln(self):
        node_labeling, _, _, _ = self._calc_wtsd()
        return node_labeling.reshape(self.img_shape)

    def get_gt_soln(self):
        node_labeling, _, _, _ = self._calc_gt_wtsd()
        return node_labeling.reshape(self.img_shape)

    def _calc_gt_wtsd(self):
        return compute_mws_segmentation_cstm(self.gt_affs.ravel(), self.gt_valid_edges.ravel(), self.gt_offsets,
                                             self.gt_separating_channel, self.img_shape)

    def _calc_wtsd(self):
        return compute_mws_segmentation_cstm(self.current_affs.ravel(), self.valid_edges.ravel(), self.mtx_offsets,
                                             self.mtx_separating_channel, self.img_shape)

    def _calc_prtl_wtsd(self):
        pass

    def _update_state(self):
        return

    def execute_action(self, action):
        return

    def reset(self):
        return

