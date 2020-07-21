import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.metrics import pairwise_distances
from skimage.feature import canny
from scipy.ndimage import maximum_filter
import numpy as np


class FullySupervisedReward(object):

    def __init__(self, env):
        super(FullySupervisedReward, self).__init__()
        self.env = env

    def get(self, diff=None, actions=None, res_seg=None):
        if self.env.discrete_action_space:
            new_diff = diff - (self.env.state[0].float() - self.env.gt_edge_weights).abs()
            reward = -(new_diff < -0.05).float() * 0.5 + (new_diff > 0.05).float()
            reward -= (((self.env.state[0].float() - self.env.gt_edge_weights).abs() > 0.1) & (actions == 0)).float() * 0.5  # penalize 0 actions when edge still different from gt
            reward += (((self.env.state[0].float() - self.env.gt_edge_weights).abs() < 0.1) & (actions == 0)).float()
        else:
            # new_diff = diff - (self.env.state[0] - self.env.gt_edge_weights).abs()
            # reward = (new_diff > 0).float() * 0.8 - (new_diff < 0).float() * 0.2
            # gt_diff = (actions - self.env.b_gt_edge_weights).abs()
            gt_diff = (actions - self.env.sg_gt_edge_weights).abs()
            # gt_diff = (actions - self.env.gt_edge_weights).abs()
            # pos_rew = (gt_diff < 0.2).float()
            # favor_separations = self.env.gt_edge_weights * actions
            # reward = 1-gt_diff


            # reward = ((gt_diff <= 0.2).float() + (gt_diff <= 0.1).float() + (gt_diff <= 0.01).float()) / 10

            reward = -gt_diff
            reward = reward + (gt_diff <= 0.05).float()

        return reward

    # reward = (new_diff > 0).float() * 1 - (new_diff < 0).float() * 2
    # reward -= (((self.env.state[0] - self.env.gt_edge_weights).abs() > 0.1) & (
    #             actions == 0)).float() * 2  # penalize 0 actions when edge still different from gt


class GlobalSparseReward():
    def __init__(self, env):
        self.env = env

    def get(self, diff=None, actions=None, res_seg=None):
        qual = diff.abs().sum()
        if qual <= self.env.stop_quality:
            return torch.tensor([1.0])
        else:
            return torch.tensor([-0.1])


class UnSupervisedReward(object):

    def __init__(self, env):
        super(UnSupervisedReward, self).__init__()
        self.env = env

    def get(self, diff=None, actions=None, res_seg=None):
        return -torch.ones_like(self.env.state[0])


class ObjectLevelReward(object):

    def __init__(self, env):
        super(ObjectLevelReward, self).__init__()
        self.env = env

    def get(self, diff=None, actions=None, res_seg=None):
        res_seg += 1
        gt = self.env.gt_seg + 1
        edge_ids = self.env.edge_ids[:, :self.env.edge_ids.shape[1]//2]
        reward = torch.zeros(edge_ids.shape[1])
        for obj in torch.unique(res_seg):
            mask = (obj == res_seg).float()
            dependant_sp = torch.unique(self.env.init_sp_seg.cpu() * mask)[1:]

            masked_gt = gt * mask
            gt_objs = torch.unique(masked_gt)[1:]
            diff_n_obj = 1 - len(gt_objs)  # should be zero if only two vals in mask (0 und obj)

            rel_gt_overlap, rel_seg_overlap = 0, 0
            for gt_obj in gt_objs:
                obj_mass = (masked_gt == gt_obj).float().sum()
                rel_gt_overlap += 1 - (obj_mass / (gt == gt_obj).float().sum())
                rel_seg_overlap += 1 - (obj_mass / mask.float().sum())

                # store all edge ids that at least one depenD_SP as incidental node
            edge_indices = torch.empty(0).long()
            for sp in dependant_sp:
                edge_indices = torch.cat((edge_indices, ((sp - 1).long() == edge_ids[0].cpu()).nonzero()))
                edge_indices = torch.cat((edge_indices, ((sp - 1).long() == edge_ids[1].cpu()).nonzero()))

            reward[edge_indices.squeeze()] = - rel_gt_overlap - rel_seg_overlap + diff_n_obj

        return reward


class DiceReward(object):
    # TODO
    def __init__(self, env):
        super(DiceReward, self).__init__()
        self.env = env

    def get(self, diff=None, actions=None, res_seg=None):
        pass


class GraphDiceReward(object):

    def __init__(self, env):
        super(GraphDiceReward, self).__init__()
        self.epsilon = 1
        self.env = env
        self.class_weights = torch.tensor([1.0, 1.0])
        self.reward_offset = torch.tensor([[-0.7], [-0.7]])

    def get(self, diff=None, actions=None, res_seg=None):
        # compute per channel Dice Coefficient
        input = torch.stack([1-self.env.state[0], self.env.state[0]], 0)
        target = torch.stack([self.env.gt_edge_weights == 0, self.env.gt_edge_weights == 1], 0).float()
        intersect = (input * target).sum(-1)

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        dice_score = 2 * (intersect / denominator.clamp(min=self.epsilon))
        dice_score = dice_score * self.class_weights.to(dice_score.device)

        reward = ((torch.ones_like(target) * dice_score.sum()) + self.reward_offset.to(dice_score.device)).sum(0)
        return reward


class SubGraphDiceReward(object):

    def __init__(self, env):
        super(SubGraphDiceReward, self).__init__()
        self.epsilon = 1e-6
        self.env = env
        self.class_weights = torch.tensor([1.0, 1.0]).unsqueeze(-1)
        self.reward_offset = torch.tensor([[-0.7], [-0.7]]).unsqueeze(-1)

    def get(self, diff=None, actions=None, res_seg=None):
        # compute per channel Dice Coefficient
        input = torch.stack([1-actions, actions], 0)
        target = torch.stack([self.env.sg_gt_edge_weights == 0, self.env.sg_gt_edge_weights == 1], 0).float()
        intersect = (input * target).sum(-1)

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        dice_score = 2 * (intersect / denominator.clamp(min=self.epsilon))
        dice_score = dice_score * self.class_weights.to(dice_score.device)

        reward = dice_score.sum(0) - 0.5
        return reward


class FocalReward(object):
    #  https://arxiv.org/pdf/1708.02002.pdf
    def __init__(self, env):
        super(FocalReward, self).__init__()
        self.epsilon = 1
        self.env = env
        self.alpha = 0.75
        self.gamma = 4
        self.offset = 10
        self.eps = 1e-10

    def get(self, diff=None, actions=None, res_seg=None):
        target = torch.stack([self.env.gt_edge_weights == 0, self.env.gt_edge_weights == 1], 0).float()

        p_t = (1-self.env.state[0]) * target[0] + self.env.state[0] * target[1]
        fl = self.alpha * (1 - p_t) ** self.gamma * (torch.log(p_t + self.eps))

        reward = fl + self.offset
        return reward


class GraphDiceLoss(nn.Module):

    def __init__(self):
        super(GraphDiceLoss, self).__init__()
        self.epsilon = 1
        self.class_weights = torch.tensor([0.5, 0.5])

    def forward(self, pred, tgt):
        # compute per channel Dice Coefficient
        input = torch.stack([1-pred, pred], 0)
        target = torch.stack([tgt == 0, tgt == 1], 0).float()
        intersect = (input * target).sum(-1)

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        dice_score = 2 * (intersect / denominator.clamp(min=self.epsilon))
        dice_score = (dice_score * self.class_weights.to(dice_score.device)).sum()
        return 1 - dice_score


class HoughCircles(object):
    def __init__(self, env, range_num, range_rad, min_hough_confidence):
        self.range_num = range_num
        self.range_rad = range_rad
        self.n_samples = 12
        self.min_hough_confidence = min_hough_confidence
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.env = env
        angles = torch.linspace(0, 2 * np.pi, self.n_samples)
        self.angle_scals = torch.stack([np.cos(angles), np.sin(angles)], dim=1)
        self.circle_thresh = 0.85

    def get(self, actions=None, res_segs=None):
        import timeit
        import matplotlib.pyplot as plt

        start = timeit.timeit()


        edge_reward = torch.zeros_like(actions)
        for g_idx, res_seg in enumerate(res_segs):
            edges = self.env.b_edge_ids[:, self.env.e_offs[g_idx]:self.env.e_offs[g_idx+1]] - self.env.n_offs[g_idx]
            segmentation = (res_seg / res_seg.max())
            obj_lbls = torch.unique(segmentation)
            masses = torch.zeros(obj_lbls.size(), dtype=torch.float)
            # centers_of_mass = np.zeros(obj_lbls.size() + (2, ), dtype=np.float)
            involved_bad_sp = []

            for idx, obj_lbl in enumerate(obj_lbls):
                mask = (segmentation == obj_lbl).float()
                masses[idx] = mask.sum().item()
                # nonzeros = torch.nonzero(mask)
                # centers_of_mass[idx] = (nonzeros.sum(0).float() / masses[idx]).cpu().numpy()
                involved_bad_sp.append(torch.unique(mask * (self.env.init_sp_seg[g_idx] + 1))[1:] - 1)

            bg_mass, bg_ind = torch.max(masses, dim=0)
            bg_sp = involved_bad_sp[bg_ind]
            if bg_ind < len(involved_bad_sp) - 1:
                involved_bad_sp = torch.unique(torch.cat(involved_bad_sp[:bg_ind] + involved_bad_sp[bg_ind+1:]))
            else:
                if len(involved_bad_sp) > 1:
                    involved_bad_sp = torch.unique(torch.cat(involved_bad_sp[:bg_ind]))
                else:  # no need to search for circles if only background
                    mask_bg_sp = ((edges.view(-1).float().unsqueeze(-1) == bg_sp.unsqueeze(0)).sum(-1) == 1).view(2, -1)
                    mask_bg_sp = mask_bg_sp[0] | mask_bg_sp[1]
                    edge_reward[torch.nonzero(mask_bg_sp).squeeze() + self.env.e_offs[g_idx]] -= (self.range_num[0] - 1) / len(obj_lbls)
                    continue

            mask_bad_sp = ((edges.view(-1).float().unsqueeze(-1) == involved_bad_sp.unsqueeze(0)).sum(-1) == 1).view(2, -1)
            mask_bad_sp = mask_bad_sp[0] | mask_bad_sp[1]
            ind_bad_sp = torch.nonzero(mask_bad_sp).squeeze() + self.env.e_offs[g_idx]

            if len(obj_lbls) < self.range_num[0] - 1:
                mask_bg_sp = ((edges.view(-1).float().unsqueeze(-1) == bg_sp.unsqueeze(0)).sum(-1) == 1).view(2, -1)
                mask_bg_sp = mask_bg_sp[0] | mask_bg_sp[1]
                edge_reward[torch.nonzero(mask_bg_sp).squeeze() + self.env.e_offs[g_idx]] -= (self.range_num[0] - 1) / len(obj_lbls)
                edge_reward[ind_bad_sp] = 0
                continue
            elif len(obj_lbls) > self.range_num[1] + 1:
                edge_reward[ind_bad_sp] -= len(obj_lbls) / (self.range_num[1] + 1)
                continue

            edge_image = ((- self.max_p(-segmentation.unsqueeze(0)).squeeze()) != segmentation).float().cpu().numpy()
            # edges0 = (imax | imin).float().cpu().numpy()
            # end = timeit.timeit()
            # maxfil_time = end - start

            # Detect two radii
            hough_radii = np.arange(self.range_rad[0], self.range_rad[1]) - 1  # account for min filter (inner circle edge)
            hough_res = hough_circle(edge_image, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=self.range_num[1])
            mp_circles = torch.from_numpy(np.stack([cx, cy], axis=1))
            accepted_circles = (cx > (self.range_rad[1] - 1)) & (cx < segmentation.shape[0] - (self.range_rad[1] - 1)) & \
                               (cy > (self.range_rad[1] - 1)) & (cy < segmentation.shape[1] - (self.range_rad[1] - 1))
            # accepted_circles = accums > self.circle_thresh

            if any(accepted_circles):
                mp_circles = mp_circles[accepted_circles]

                sample_r = torch.randint(self.range_rad[0] // 2, self.range_rad[1] - 1, (self.n_samples,))
                points = self.angle_scals * sample_r.unsqueeze(1).float()
                sample_points = (points.unsqueeze(0) + mp_circles.float().unsqueeze(1)).long()

                assert all(sample_points.view(-1) >= 0) and all(sample_points.view(-1) < self.env.init_sp_seg.shape[1])

                circle_sp = self.env.init_sp_seg[g_idx, sample_points[..., 0], sample_points[..., 1]]

                mask_circle_sp = (edges.float().unsqueeze(-1).unsqueeze(-1) == circle_sp.unsqueeze(0).unsqueeze(0)).sum(-1) >= 1
                mask_circle_sps = mask_circle_sp[0] | mask_circle_sp[1]

                for mask_circle_sp, val in zip(mask_circle_sps.T, accums):
                    circle_edges = torch.nonzero(mask_circle_sp).squeeze() + self.env.e_offs[g_idx]
                    edge_reward[circle_edges] += val * 10

        reward = edge_reward[self.env.b_subgraph_indices].view(-1, self.env.args.s_subgraph).sum(-1)

        end = timeit.timeit()
        time = start - end

        return reward


class HoughCircles_lg(object):
    def __init__(self, env, range_num, range_rad, min_hough_confidence):
        self.range_num = range_num
        self.range_rad = range_rad
        self.n_samples = 12
        self.min_hough_confidence = min_hough_confidence
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.env = env
        angles = torch.linspace(0, 2 * np.pi, self.n_samples)
        self.angle_scals = torch.stack([np.cos(angles), np.sin(angles)], dim=1)
        self.circle_thresh = 0.85

    def get(self, actions=None, res_segs=None):
        import timeit
        import matplotlib.pyplot as plt

        start = timeit.timeit()

        edge_reward_g = torch.ones(res_segs.shape[0])
        edge_reward_l = torch.zeros_like(actions)

        for g_idx, res_seg in enumerate(res_segs):
            edges = self.env.b_edge_ids[:, self.env.e_offs[g_idx]:self.env.e_offs[g_idx+1]] - self.env.n_offs[g_idx]
            segmentation = (res_seg / res_seg.max())
            obj_lbls = torch.unique(segmentation)
            masses = torch.zeros(obj_lbls.size(), dtype=torch.float)
            # centers_of_mass = np.zeros(obj_lbls.size() + (2, ), dtype=np.float)
            involved_bad_sp = []

            if len(obj_lbls) < self.range_num[0] - 1:
                edge_reward_g[g_idx] = - (self.range_num[0] - 1) / len(obj_lbls)
                continue
            elif len(obj_lbls) > self.range_num[1] + 1:
                edge_reward_g[g_idx] = - len(obj_lbls) / (self.range_num[1] + 1)
                continue

            edge_image = ((- self.max_p(-segmentation.unsqueeze(0)).squeeze()) != segmentation).float().cpu().numpy()

            hough_radii = np.arange(self.range_rad[0], self.range_rad[1]) - 1  # account for min filter (inner circle edge)
            hough_res = hough_circle(edge_image, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=self.range_num[1])
            mp_circles = torch.from_numpy(np.stack([cx, cy], axis=1))
            accepted_circles = (cx > (self.range_rad[1] - 1)) & (cx < segmentation.shape[0] - (self.range_rad[1] - 1)) & \
                               (cy > (self.range_rad[1] - 1)) & (cy < segmentation.shape[1] - (self.range_rad[1] - 1))

            if any(accepted_circles):
                mp_circles = mp_circles[accepted_circles]

                sample_r = torch.randint(self.range_rad[0] // 2, self.range_rad[1] - 1, (self.n_samples,))
                points = self.angle_scals * sample_r.unsqueeze(1).float()
                sample_points = (points.unsqueeze(0) + mp_circles.float().unsqueeze(1)).long()

                assert all(sample_points.view(-1) >= 0) and all(sample_points.view(-1) < self.env.init_sp_seg.shape[1])

                circle_sp = self.env.init_sp_seg[g_idx, sample_points[..., 0], sample_points[..., 1]]

                mask_circle_sp = (edges.float().unsqueeze(-1).unsqueeze(-1) == circle_sp.unsqueeze(0).unsqueeze(0)).sum(-1) >= 1
                mask_circle_sps = mask_circle_sp[0] | mask_circle_sp[1]

                for mask_circle_sp, val in zip(mask_circle_sps.T, accums):
                    circle_edges = torch.nonzero(mask_circle_sp).squeeze() + self.env.e_offs[g_idx]
                    edge_reward_l[circle_edges] += val * 10

        reward_l = edge_reward_l[self.env.b_subgraph_indices].view(-1, self.env.args.s_subgraph).sum(-1)

        end = timeit.timeit()
        time = start - end

        return reward_l.to(actions.device), edge_reward_g.to(actions.device)
