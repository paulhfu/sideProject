from environments.environment_bc import Environment
from mutex_watershed import compute_mws_segmentation_cstm
from affogato.segmentation.mws import get_valid_edges
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import inf


class MtxWtsdEnvBc(Environment):
    def __init__(self, affs, separating_channel, offsets, strides, win_reward, gt_affinities=None,
                 stop_cnt=15):
        super(MtxWtsdEnvBc, self).__init__(stop_cnt)

        self.initial_affs = affs
        self.current_affs = self.initial_affs.copy()
        self.mtx_separating_channel = separating_channel
        self.img_shape = affs[0].shape
        self.mtx_offsets = offsets
        self.mtx_strides = strides
        self.valid_edges = get_valid_edges((len(self.mtx_offsets),) + self.img_shape, self.mtx_offsets,
                                           self.mtx_separating_channel, self.mtx_strides, False)
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

    def show_current_soln(self):
        node_labeling, _, _, _ = self._calc_wtsd()
        node_labeling = node_labeling.reshape(self.img_shape)
        seg = cm.prism(node_labeling / node_labeling.max())
        plt.imshow(seg)
        plt.show()

    def get_current_soln(self):
        node_labeling = self._calc_wtsd()
        return node_labeling.reshape(self.img_shape)

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

