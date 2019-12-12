from ril_function_models import DN_DQN
from mutex_watershed import compute_partial_mws_prim_segmentation, compute_mws_prim_segmentation
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation.mws import get_valid_edges
from sklearn.metrics import mean_squared_error
from replayMemory import ReplayMemory, Transition
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import os

writer = SummaryWriter(logdir='./logs')

class Agent(object):
    def __init__(self, alpha, gamma, n_state_channels, n_mtxs, n_actions, n_sel_mtxs, device,
                 eps=0.9, eps_min=0.000001, replace=2):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.init_eps = eps
        self.eps = self.init_eps
        self.eps_min = eps_min

        self.q_eval = DN_DQN(num_classes=n_actions+n_sel_mtxs, num_inchannels=n_state_channels, device=device)
        self.q_next = DN_DQN(num_classes=n_actions+n_sel_mtxs, num_inchannels=n_state_channels, device=device)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = n_actions
        self.n_mtxs = n_mtxs
        self.n_sel_mtxs = n_sel_mtxs
        self.n_state_channels = n_state_channels
        self.mem = ReplayMemory(capacity=8)
        self.learn_steps = 0
        self.steps = 0
        self.replace_tgt_cnt = replace

    def reset_eps(self, eps):
        self.eps = eps

    def safe_model(self, dir):
        torch.save(self.q_eval.state_dict(), os.path.join(dir, 'q_eval_func'))
        # torch.save(self.q_next.state_dict(), os.path.join(dir, 'q_next_func'))

    def load_model(self, dir):
        self.q_eval.load_state_dict(torch.load(os.path.join(dir, 'q_eval_func')), strict=True)
        self.q_next.load_state_dict(torch.load(os.path.join(dir, 'q_eval_func')), strict=True)

    def store_transit(self, state, action, reward, next_state):
        # the arguments should all be gpu tensors with 0 grads
        # print(f'action: {action}    reward {reward}')
        self.mem.push(state, action, reward, next_state)

    def get_action(self, state):
        if np.random.random() < (1-self.eps):
            self.q_eval.eval()
            q_vals = self.q_eval(torch.tensor(state, dtype=torch.float32).to(self.q_eval.device).unsqueeze(0)).squeeze()
            action = [torch.argmax(q_vals[:self.n_sel_mtxs], dim=0).item(),
                      torch.argmax(q_vals[self.n_sel_mtxs:], dim=0).item()]
        else:
            action = [np.random.randint(0, self.n_sel_mtxs),
                      np.random.randint(0, self.n_actions)]
        self.steps += 1
        return action

    def learn(self, batch_size):
        with torch.set_grad_enabled(True):
            self.q_eval.optimizer.zero_grad()
            if self.replace_tgt_cnt is not None and self.learn_steps % self.replace_tgt_cnt == 0:
                self.q_next.load_state_dict(self.q_eval.state_dict())
                # print(f'eps: {self.eps}')
                if self.eps - 1e-3 > self.eps_min:
                    self.eps -= 1e-3
                else:
                    self.eps = self.eps_min

            batch = self.mem.sample(batch_size)
            batch = Transition(*zip(*batch))
            state = torch.tensor(list(batch.state), dtype=torch.float32).to(self.q_eval.device)
            next_state = torch.tensor(list(batch.next_state), dtype=torch.float32).to(self.q_next.device)
            actions = torch.tensor(list(batch.action), dtype=torch.long).to(self.q_eval.device)
            actions[:, 1] += self.n_sel_mtxs
            rewards = torch.tensor(list(batch.reward), dtype=torch.float32).to(self.q_eval.device)

            predQ = self.q_eval(state).gather(1, actions)
            nextQ = self.q_next(next_state)
            nextQ = torch.cat([nextQ[:, :self.n_sel_mtxs].max(1)[0].detach().unsqueeze(1),
                               nextQ[:, self.n_sel_mtxs:].max(1)[0].detach().unsqueeze(1)], dim=1)

            tgtQ = (nextQ * self.gamma) + rewards.unsqueeze(1)

            # # pred_action_val = pred_action_val.view(batch_size, -1)
            # tgtQ = predQ.clone()
            # tgtQ = tgtQ.view(*shape)\
            #     .transpose(0, 2)\
            #     .transpose(1, 2)\
            #     .contiguous()
            # mem_shape = tgtQ.shape
            # tgtQ = tgtQ.view(3, -1)
            # tgtQ[action_] = (rewards.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.gamma * pred_action_val).view(-1)
            #
            # tgtQ = tgtQ.view(*mem_shape).transpose(1, 2).transpose(0, 2).contiguous()
            # pred_action = pred_action.view(2, 2, 4, 4)
            # tgtQ = tgtQ.view(batch_size, self.n_mtxs, self.n_actions, nextQ.shape[-2], nextQ.shape[-1])
            # predQ = predQ.view(batch_size, self.n_mtxs, self.n_actions, nextQ.shape[-2], nextQ.shape[-1])

            # be sure we constructed our Q values correctly
            # assert np.all((torch.argmax(tgtQ != predQ, dim=2) == pred_action).detach().cpu().numpy().astype(np.bool).ravel())

            loss = self.q_eval.loss(tgtQ, predQ)
            writer.add_scalar("step/loss", loss.item(), self.learn_steps)
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            for param in self.q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.q_eval.optimizer.step()
            self.learn_steps += 1

class Mtx_wtsd_env(object):
    def __init__(self, affs, separating_channel, separating_action, offsets, strides, gt_affinities, use_bbox=False):
        super(Mtx_wtsd_env, self).__init__()
        self.initial_affs = affs
        self.all_gt_affs = gt_affinities

        self.done = False
        self.use_bbox = False
        self.action_agression = 2
        self.trust_data = 15
        self.separating_action = separating_action
        self.current_affs = self.initial_affs.copy()
        self.sorted_mtxs = [[w, id] for id, w in zip(range(len(self.current_affs[:separating_channel].ravel())), self.current_affs[:separating_channel].ravel())]
        self.sorted_attr = [[w, id] for id, w in zip(range(len(self.current_affs[:separating_channel].ravel())), self.current_affs[separating_channel:].ravel())]
        self.sorted_attr.sort()
        self.sorted_mtxs.sort()
        self.img_shape = tuple(affs[0].shape)
        self.mtx_offsets = offsets
        self.mtx_strides = strides
        self.mtx_separating_channel = separating_channel
        self.valid_edges = get_valid_edges((len(self.mtx_offsets),) + self.img_shape, self.mtx_offsets, self.mtx_separating_channel, self.mtx_strides, False)
        self.gt_affinities = (gt_affinities[:separating_channel] == 0).astype(np.float) * self.valid_edges[:2]
        self.bbox = [0, 0]
        self.reward_func = mean_squared_error
        self.mtx_wtsd_start_iter = 10
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self.quality = 8
        self.counter = 0
        self.iteration = 0
        self.last_reward = -100
        self.accumulated_reward = 0
        self.stop_reward = -200
        self._update_state()

    def show_current_soln(self):
        complete_labeling = compute_mws_prim_segmentation(self.current_affs.ravel(),
                                                            self.valid_edges.ravel(),
                                                            self.mtx_offsets,
                                                            self.mtx_separating_channel,
                                                            self.img_shape)
        plt.imshow(complete_labeling.reshape(self.img_shape));
        plt.show()

    def _update_state(self):
        # complete_labeling = compute_mws_prim_segmentation(self.current_affs.ravel(),
        #                                                     self.valid_edges.ravel(),
        #                                                     self.mtx_offsets,
        #                                                     self.mtx_separating_channel,
        #                                                     self.img_shape)
        # gt_labeling = compute_mws_segmentation(self.current_affs, self.mtx_offsets, self.mtx_separating_channel, algorithm='prim')
        node_labeling, cut_edges, used_mtxs = compute_partial_mws_prim_segmentation(self.current_affs.ravel(),
                                                                                    self.valid_edges.ravel(),
                                                                                    self.mtx_offsets,
                                                                                    self.mtx_separating_channel,
                                                                                    self.img_shape, self.mtx_wtsd_max_iter)
        # assert all(gt_labeling.ravel() == node_labeling-1)
        # plt.imshow(complete_labeling.reshape(self.img_shape));
        # plt.show()
        # print(self.current_affs - self.initial_affs)
        # self.show_current_soln()
        cut_edge_imgs = np.zeros(node_labeling.size*len(self.mtx_offsets), dtype=np.float)
        for edge_id in cut_edges + used_mtxs:
            cut_edge_imgs[edge_id] = 1
        cut_edge_imgs = cut_edge_imgs.reshape((len(self.mtx_offsets),)+self.img_shape)
        if self.use_bbox:
            ymax_vals, xmax_vals = self._bbox(cut_edge_imgs)
            self.bbox = [np.max(ymax_vals), np.max(xmax_vals)]
        else:
            self.bbox = cut_edge_imgs.shape[1:]
        self.state = np.concatenate((cut_edge_imgs[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]],
                                     self.current_affs[:, 0:self.bbox[0], 0:self.bbox[1]]),
                                    axis=0).astype(np.float)

        if self.counter % 20 == 0:
            # print(self.current_affs - self.initial_affs)
            node_labeling = node_labeling.reshape(self.img_shape)
            writer.add_image('res/igmRes', cm.prism(node_labeling / node_labeling.max()), self.counter)
            # self.writer.add_image('res/imgInit', cm.prism(complete_labeling.reshape(self.img_shape) / complete_labeling.reshape(self.img_shape).max()), self.counter)
        # if self.counter == 1:
        #     complete_labeling = complete_labeling.reshape(self.img_shape)
        #     show_complete_seg_gt = cm.prism(gt_labeling / gt_labeling.max())
        #     show_complete_seg = cm.prism(complete_labeling / complete_labeling.max())
        #     img = np.concatenate([show_complete_seg_gt, show_complete_seg], axis=1)
        #     plt.imshow(img);plt.show()
        #     self.counter = 0

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
        num_actions = 0
        last_affs = self.current_affs.copy()
        if action[0] < self.separating_action:
            edge_id = self.sorted_attr[action[0]][1]
            c = edge_id // (self.current_affs.shape[2] * self.current_affs.shape[1])

            iv = edge_id % (self.current_affs.shape[2]*self.current_affs.shape[1])
            y = iv // self.current_affs.shape[2]
            x = iv % self.current_affs.shape[2]
            if action[1] == 0:
                num_actions = 1
                self.current_affs[c, y, x] /= self.action_agression
                self.sorted_attr[action[0]][0] /= self.action_agression
                i = action[0]
                while i > 0 and self.sorted_attr[i][0] < self.sorted_attr[i-1][0]:
                    self.sorted_attr[i], self.sorted_attr[i-1] = self.sorted_attr[i-1], self.sorted_attr[i]
                    i -= 1
            if action[1] == 1:
                num_actions = 1
                self.current_affs[c, y, x] = max(1, self.current_affs[c, y, x]*self.action_agression)
                self.sorted_attr[action[0]][0] = max(1, self.sorted_attr[action[0]][0]*self.action_agression)
                i = action[0]
                while i < len(self.sorted_attr)-1 and self.sorted_attr[i][0] > self.sorted_attr[i+1][0]:
                    self.sorted_attr[i], self.sorted_attr[i+1] = self.sorted_attr[i+1], self.sorted_attr[i]
                    i += 1
        else:
            edge_id = self.sorted_mtxs[action[0] - self.separating_action][1]
            c = edge_id // (self.current_affs.shape[2]*self.current_affs.shape[1]) + self.mtx_separating_channel

            iv = edge_id % (self.current_affs.shape[2]*self.current_affs.shape[1])
            y = iv // self.current_affs.shape[2]
            x = iv % self.current_affs.shape[2]
            if action[1] == 0:
                num_actions = 1
                self.current_affs[c, y, x] /= self.action_agression
                self.sorted_mtxs[action[0]][0] /= self.action_agression
                i = action[0]
                while i > 0 and self.sorted_mtxs[i][0] < self.sorted_mtxs[i-1][0]:
                    self.sorted_mtxs[i], self.sorted_mtxs[i-1] = self.sorted_mtxs[i-1], self.sorted_mtxs[i]
                    i -= 1
            if action[1] == 1:
                num_actions = 1
                self.current_affs[c, y, x] = max(1, self.current_affs[c, y, x]*self.action_agression)
                self.sorted_mtxs[action[0]][0] = max(1, self.sorted_mtxs[action[0]][0]*self.action_agression)
                i = action[0]
                while i < len(self.sorted_mtxs)-1 and self.sorted_mtxs[i][0] > self.sorted_mtxs[i+1][0]:
                    self.sorted_mtxs[i], self.sorted_mtxs[i+1] = self.sorted_mtxs[i+1], self.sorted_mtxs[i]
                    i += 1

        # for idx, channel_actions in enumerate(actions):
        #     mask = np.array((channel_actions == 1) / 2 + (channel_actions == 2) * 2, dtype=np.float)
        #     mask_inv = np.array(mask == 0, dtype=np.float)
        #     num_actions += np.sum(mask != 0)
        #     self.current_affs[self.mtx_separating_channel+idx, 0:self.bbox[0], 0:self.bbox[1]] = \
        #         self.current_affs[self.mtx_separating_channel+idx, 0:self.bbox[0], 0:self.bbox[1]] * mask + \
        #         self.current_affs[self.mtx_separating_channel+idx, 0:self.bbox[0], 0:self.bbox[1]] * mask_inv
        self._update_state()
        data_changed = np.sum(np.abs(self.current_affs - self.initial_affs))
        penalize_change = 0
        if data_changed > self.trust_data:
            penalize_change = self.trust_data - data_changed

        reward = - 10 + penalize_change + \
                 np.sum(self.state[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]].ravel()*2-1 ==
                                            self.gt_affinities[:, 0:self.bbox[0], 0:self.bbox[1]].ravel()) - \
                 np.sum(self.state[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]].ravel() !=
                                            self.gt_affinities[:, 0:self.bbox[0], 0:self.bbox[1]].ravel())

        if reward + 10 >= 7:
            reward = 100
            self.done = True
            self.iteration += 1
        if reward < self.last_reward:
            a=1
            self.current_affs = last_affs.copy()
        self.last_reward = reward

        writer.add_scalar("step/reward", reward, self.counter*self.iteration)
        self.counter += 1
        self.accumulated_reward += reward
        # print(f"reward: {reward}")
        if self.counter > 15:
            reward = -100
            self.done = True
            self.iteration += 1
        return self.state, reward

    def reset(self):
        self.current_affs = self.initial_affs.copy()
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self.done = False
        self.accumulated_reward = 0
        self._update_state()
        self.counter = 0
