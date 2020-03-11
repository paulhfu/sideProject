from mutex_watershed import compute_mws_segmentation_cstm
from matplotlib import cm
import numpy as np
import torch
from utils.utils import ind_flat_2_spat
from math import inf
from environments.mutex_wtsd_bc import MtxWtsdEnvBc


class MtxWtsdEnvUnet(MtxWtsdEnvBc):
    def __init__(self, affinities, separating_channel, offsets, strides, gt_affinities, stop_cnt=500,
                 win_reward=-100, use_bbox=False, writer=None, action_aggression=2, penalize_diff_thresh=2000,
                 keep_first_state=True):
        super(MtxWtsdEnvUnet, self).__init__(affinities, separating_channel, offsets, strides, gt_affinities=gt_affinities,
                                           stop_cnt=stop_cnt, win_reward=win_reward)
        self.writer = writer
        self.gt_seg, _, _, _ = compute_mws_segmentation_cstm(gt_affinities, self.valid_edges,
                                                             self.mtx_offsets, self.mtx_separating_channel,
                                                             self.img_shape)
        self.use_bbox = use_bbox
        self.action_aggression = action_aggression
        self.penalize_diff_thresh = penalize_diff_thresh
        self.loosing_diff_thresh = penalize_diff_thresh * 3

        self.bbox = np.array(self.img_shape)
        self.keep_first_state = keep_first_state
        self._update_state_1()

    def update_data(self, affinities, gt_affinities=None):
        super(MtxWtsdEnvUnet, self).update_data(affinities, gt_affinities)
        self.gt_seg, _, _, _ = self._calc_gt_wtsd()

    def _get_neighbors_features(self):
        node_labeling, cut_edges, used_mtxs, neighbors_features = self._calc_wtsd()

    def _update_state_1(self, actions=None):
        #update state for the retrace algorithm
        node_labeling, _, cut_edges, used_mtxs = self._calc_wtsd()
        # used_mtxs = used_mtxs - self.mtx_separating_channel * node_labeling.size  # remove offset

        self.used_edges_mask = torch.zeros(node_labeling.size*len(self.mtx_offsets), dtype=torch.bool)
        for edge_id in used_mtxs + cut_edges:
            self.used_edges_mask[edge_id.astype(np.long)] = True

        self.used_edges_mask = self.used_edges_mask.reshape((len(self.mtx_offsets),)+self.img_shape)
        self.state = self.current_affs

        if self.counter == 0:
            # node_labeling = node_labeling.reshape(self.img_shape)
            # import matplotlib.pyplot as plt;plt.imshow(cm.prism(node_labeling / node_labeling.max()));plt.show();
            if self.writer is not None:
                self.writer.add_image('res/igmRes', cm.prism(node_labeling / node_labeling.max()), self.counter)

        ret = None
        if actions is not None:
            ret = self.calculate_reward_1(actions)
        return ret

    def _update_state(self):
        node_labeling, neighbors, cutting_edges, mutexes = self._calc_wtsd()

        used_edge_imgs = torch.zeros(node_labeling.size*len(self.mtx_offsets), dtype=torch.bool)

        for edge_id in cutting_edges + mutexes:
            used_edge_imgs[edge_id] = 1

        self.used_edges_mask = used_edge_imgs.reshape((len(self.mtx_offsets),)+self.img_shape)
        self.state = np.concatenate((self.used_edges_mask, self.current_affs), axis=0).astype(np.float)
        self.mtx_wtsd_max_iter += self.mtx_wtsd_iter_step

        if self.counter == 0:
            # node_labeling = node_labeling.reshape(self.img_shape)
            # import matplotlib.pyplot as plt;plt.imshow(cm.prism(node_labeling / node_labeling.max()));plt.show();
            if self.writer is not None:
                self.writer.add_image('res/igmRes', cm.prism(node_labeling / node_labeling.max()), self.counter)

        return self.calculate_reward(neighbors, self.gt_seg, node_labeling.reshape(self.img_shape), mutexes,\
                                     cutting_edges, node_labeling.size*len(self.mtx_offsets))

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

    def calculate_reward_1(self, actions):
        # penalty on up or down action of unused mtxs, also on variance of selection in terms of datapt indices and l2norm (pca components), mtxs with up/down action that where used get reward according to ground truth
        # assuming we start from an oversegmentation and only wand to reduce mutex values. it is sufficient to only reduce mutex values that where used by watershed
        action_indices = actions != 0  # have only one action
        # components, variances = pca_svd(X=action_indices, k=1)
        # n_wrong_selected = -(torch.sum((action_indices + self.used_edges_mask).bool() != self.used_edges_mask) / action_indices.numel())
        essential_mtxs = self.used_edges_mask == action_indices
        wrong_selected = -((action_indices + self.used_edges_mask).bool() != self.used_edges_mask).float()
        gt_penalty = - essential_mtxs.float() * torch.from_numpy(self.current_affs - self.gt_affs).abs()
        return wrong_selected + gt_penalty
        


    def calculate_reward(self, neighbors, gt_seg, new_seg, mutexes, cutting_edges, num_edges):
        # penalty on up or down action of unused mtxs, also on variance of selection in terms of datapt indices and l2norm, mtxs with up/down action that where used get reward according to ground truth
        indices = np.concatenate((np.expand_dims(ind_flat_2_spat(neighbors[:, 0], gt_seg.shape), 1),
                                  np.expand_dims(ind_flat_2_spat(neighbors[:, 1], gt_seg.shape), 1)), axis=1)
        rewards = np.zeros(num_edges)
        for neighbor, mtxs, ces in zip(indices, mutexes, cutting_edges):
            mask_n1, mask_n2 = new_seg == new_seg[neighbor[0, 0], neighbor[0, 1]], new_seg == new_seg[neighbor[1, 0], neighbor[1, 1]]
            mask = mask_n1 + mask_n2
            # import matplotlib.pyplot as plt; plt.imshow(mask); plt.show();plt.imshow(cm.prism(new_seg / new_seg.max()));plt.show();plt.imshow(gt_seg);plt.show();
            mskd_gt_seg = mask * gt_seg
            mskd_new_seg = mask * new_seg
            n_obj_gt = np.unique(mskd_gt_seg)
            n_obj_new = np.unique(mskd_new_seg)
            n_obj_gt = n_obj_gt[1:] if n_obj_gt[0] == 0 else n_obj_gt
            n_obj_new = n_obj_new[1:] if n_obj_new[0] == 0 else n_obj_new
            n_obj_pnlty = - abs(len(n_obj_new) - len(n_obj_gt)) * 10
            overlap_pnlty = 0
            if n_obj_pnlty == 0:
                assert len(n_obj_new) == 2
                mask_gt1, mask_gt2 = mskd_gt_seg == n_obj_gt[0], mskd_gt_seg == n_obj_gt[1]
                overlap_gt1_n1, overlap_gt1_n2 = np.sum(mask_gt1 * mask_n1) / np.sum(mask_n1), \
                                                 np.sum(mask_gt1 * mask_n2) / np.sum(mask_n2)
                if overlap_gt1_n1 < overlap_gt1_n2:
                    mask_n2, mask_n1 = mask_n1, mask_n2
                overlap_pnlty = (np.sum(mask_gt1 * mask_n1) + np.sum(mask_gt2 * mask_n2) - np.sum(mask)) / np.sum(mask)
            rewards[np.concatenate((mtxs, ces), axis=0)] += overlap_pnlty + n_obj_pnlty
            # img1 = np.concatenate([np.concatenate([cm.prism(new_seg / new_seg.max()), cm.prism(mask / mask.max())], axis=1),
            #                       np.concatenate([cm.prism(mskd_gt_seg / mskd_gt_seg.max()), cm.prism(mskd_new_seg / mskd_new_seg.max())], axis=1)], axis=0)
            # import matplotlib.pyplot as plt;plt.imshow(img1);plt.show();
            # a=1
        return rewards.reshape(self.current_affs.shape)

    def execute_action(self, actions):
        last_affs = self.current_affs.copy()
        mask = (actions == 1).float() / 2
        mask += (mask == 0).float()
        self.current_affs *= mask.numpy()
        reward = self._update_state_1(actions)

        # calculate reward
        self.data_changed = np.sum(np.abs(self.current_affs - self.initial_affs))
        penalize_change = 0
        if self.data_changed > self.penalize_diff_thresh:
            penalize_change = (self.penalize_diff_thresh - self.data_changed) / np.prod(self.img_shape) * 10
        reward += (penalize_change * self.used_edges_mask.float())

        total_reward = torch.sum(reward)
        if self.keep_first_state:
            keep_old_state = True
            self.current_affs = last_affs
        else:
            keep_old_state = False

        if self.writer is not None:
            self.writer.add_scalar("step/reward", total_reward, self.counter + (self.stop_cnt * self.iteration))

        self.counter += 1

        # check if finished
        if total_reward >= self.stop_quality:
            reward += (1000 * self.used_edges_mask.float())
            self.done = True
            self.iteration += 1
            self.last_reward = -inf

        if self.data_changed > self.loosing_diff_thresh:
            reward += (-100 * self.used_edges_mask.float())
            self.done = True
            self.iteration += 1
            self.last_reward = -inf

        self.acc_reward += total_reward
        try:
            assert all(((self.used_edges_mask == 0) * reward == 0).ravel())
        except:
            pass
        return self.state, reward, keep_old_state

    def reset(self):
        self.done = False
        self.acc_reward = 0
        self.last_reward = -inf
        self.counter = 0
        self.current_affs = self.initial_affs.copy()
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self._update_state_1()

