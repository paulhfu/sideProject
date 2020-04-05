import torch
import matplotlib.pyplot as plt

class FullySupervisedReward(object):

    def __init__(self, env):
        super(FullySupervisedReward, self).__init__()
        self.env = env

    def get(self, diff=None, actions=None, res_seg=None):
        new_diff = diff - (self.env.state[0] - self.env.gt_edge_weights).abs()
        reward = (new_diff > 0).float() * 1 - (new_diff < 0).float() * 2
        reward -= (((self.env.state[0] - self.env.gt_edge_weights).abs() > 0.1) & (actions == 0)).float() * 2  # penalize 0 actions when edge still different from gt
        return reward

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

        return reward / 10

