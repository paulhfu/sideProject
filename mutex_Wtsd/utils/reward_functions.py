import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
            gt_diff = (actions - self.env.gt_edge_weights).abs()
            # pos_rew = (gt_diff < 0.2).float()
            # favor_separations = self.env.gt_edge_weights * actions
            # reward = 1-gt_diff
            reward = ((gt_diff <= 0.2).float() + (gt_diff <= 0.1).float() + (gt_diff <= 0.01).float()) / 10

            # reward = - (self.env.state[0]).abs()

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