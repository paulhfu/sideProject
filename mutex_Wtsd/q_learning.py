from ril_function_models import Q_value
from mutex_watershed import compute_partial_mws_prim_segmentation, compute_mws_prim_segmentation
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation.mws import get_valid_edges
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import os

class Agent(object):
    def __init__(self, alpha, gamma, n_state_channels, n_action_channels, n_actions,
                 eps=0.2, eps_min=0.000001, replace=2):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.init_eps = eps
        self.eps = self.init_eps
        self.eps_min = eps_min

        self.q_eval = Q_value(n_state_channels, n_action_channels, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.q_next = Q_value(n_state_channels, n_action_channels, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_action_channels = n_action_channels
        self.n_actions = n_actions
        self.n_mtxs = int(n_action_channels / n_actions)
        self.mem_log = []
        self.mem_cnt = 0
        self.mem_size = 4
        self.learn_steps = 0
        self.steps = 0
        self.replace_tgt_cnt = replace

    def reset_eps(self):
        self.eps = self.init_eps

    def safe_model(self, dir):
        torch.save(self.q_eval.state_dict(), os.path.join(dir, 'q_eval_func'))
        # torch.save(self.q_next.state_dict(), os.path.join(dir, 'q_next_func'))

    def load_model(self, dir):
        self.q_eval.load_state_dict(torch.load(os.path.join(dir, 'q_eval_func')), strict=True)
        self.q_next.load_state_dict(torch.load(os.path.join(dir, 'q_eval_func')), strict=True)

    def store_transit(self, state, action, reward, next_state):
        # the arguments should all be gpu tensors with 0 grads
        if self.mem_cnt < self.mem_size:
            self.mem_log.append([state, action, reward, next_state])
        else:
            self.mem_log[self.mem_cnt % self.mem_size] = [state, action, reward, next_state]
        self.mem_cnt += 1

    def get_action(self, state):
        shape = [self.n_mtxs, self.n_actions, state[0].shape[0], state[0].shape[1]]
        if np.random.random() < (1 - self.eps):
            self.q_eval.eval()
            q_vals = self.q_eval(torch.tensor(state, dtype=torch.float32).to(self.q_eval.device).unsqueeze(0))
            q_vals = q_vals.view(*shape)
            action = torch.argmax(q_vals, dim=1).detach().cpu().numpy().astype(np.float)
        else:
            action = np.random.randint(0, self.n_action_channels, [shape[0]]+shape[2:])
        self.steps += 1
        return action

    def learn(self, batch_size):
        with torch.set_grad_enabled(True):
            self.q_eval.optimizer.zero_grad()
            if self.replace_tgt_cnt is not None and self.learn_steps % self.replace_tgt_cnt == 0:
                self.q_next.load_state_dict(self.q_eval.state_dict())

            if (self.mem_cnt % self.mem_size) + batch_size < self.mem_size:
                mem_start = int(np.random.choice(range((self.mem_cnt % self.mem_size)+1)))
            else:
                mem_start = int(np.random.choice(range((self.mem_cnt % self.mem_size) + 2 - batch_size)))

            batch = np.array(self.mem_log[mem_start:mem_start+batch_size])
            predQ = self.q_eval(torch.tensor(batch[:, 0].tolist(), dtype=torch.float32).to(self.q_eval.device))
            nextQ = self.q_next(torch.tensor(batch[:, 3].tolist(), dtype=torch.float32).to(self.q_next.device))
            shape = [batch_size, self.n_mtxs, self.n_actions, nextQ.shape[-2], nextQ.shape[-1]]

            rewards = torch.tensor(batch[:, 2].tolist(), dtype=torch.float32).to(self.q_eval.device)
            o_predQ = predQ.view(*shape).squeeze().detach().cpu().numpy()
            nextQ = nextQ.view(*shape)
            pred_action_val, pred_action = torch.max(nextQ.squeeze(), dim=2)
            pred_action = pred_action.view(-1)
            # pred_action_val = pred_action_val.view(batch_size, -1)
            tgtQ = predQ.clone()
            tgtQ = tgtQ.view(*shape)\
                .transpose(0, 2)\
                .transpose(1, 2)\
                .contiguous()
            mem_shape = tgtQ.shape
            tgtQ = tgtQ.view(3, -1)
            tgtQ[pred_action, torch.arange(pred_action.size(0))] = (rewards.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.gamma * pred_action_val).view(-1)

            tgtQ = tgtQ.view(*mem_shape).transpose(1, 2).transpose(0,2).contiguous()
            pred_action = pred_action.view(2,2,4,4)
            tgtQ = tgtQ.view(batch_size, self.n_mtxs, self.n_actions, nextQ.shape[-2], nextQ.shape[-1])
            predQ = predQ.view(batch_size, self.n_mtxs, self.n_actions, nextQ.shape[-2], nextQ.shape[-1])

            if self.steps > 5:
                if self.eps - 1e-2 > self.eps_min:
                    self.eps -= 1e-2
                else:
                    self.eps = self.eps_min

            # be sure we constructed our Q values correctly
            assert np.all((torch.argmax(tgtQ != predQ, dim=2) == pred_action).detach().cpu().numpy().astype(np.bool).ravel())

            loss = self.q_eval.loss(tgtQ, predQ)
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_steps += 1

class Mtx_wtsd_env(object):
    def __init__(self, edge_predictor, raw_image, separating_channel, offsets, strides, gt_affinities, use_bbox=False):
        super(Mtx_wtsd_env, self).__init__()
        edge_predictor.eval()
        torch.set_grad_enabled(False)
        edge_predictor.cuda()
        affs = edge_predictor(raw_image.to(edge_predictor.device))
        self.initial_affs = affs.squeeze().detach().cpu().numpy().astype(np.float)
        self.initial_affs[separating_channel:] *= -1
        self.initial_affs[separating_channel:] += +1
        self.done = False
        self.use_bbox = False

        self.gt_affinities = gt_affinities
        self.gt_affinities *= -1
        self.gt_affinities += +1
        self.current_affs = self.initial_affs
        self.img_shape = tuple(raw_image.squeeze().shape)
        self.mtx_offsets = offsets
        self.mtx_strides = strides
        self.mtx_separating_channel = separating_channel
        self.valid_edges = get_valid_edges((len(self.mtx_offsets),) + self.img_shape, self.mtx_offsets, self.mtx_separating_channel, self.mtx_strides, False)
        self.bbox = [0, 0]
        self.reward_func = mean_squared_error
        self.mtx_wtsd_start_iter = 10
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self.quality = 1000
        self.counter = 0
        self.accumulated_reward = 0
        self.stop_reward = -200
        self._update_state()

    def _update_state(self):
        complete_labeling = compute_mws_prim_segmentation(self.current_affs.ravel(),
                                                            self.valid_edges.ravel(),
                                                            self.mtx_offsets,
                                                            self.mtx_separating_channel,
                                                            self.img_shape)
        node_labeling, cut_edges, used_mtxs = compute_partial_mws_prim_segmentation(self.current_affs.ravel(),
                                                                                    self.valid_edges.ravel(),
                                                                                    self.mtx_offsets,
                                                                                    self.mtx_separating_channel,
                                                                                    self.img_shape, self.mtx_wtsd_max_iter)
        cut_edge_imgs = np.zeros(node_labeling.size*len(self.mtx_offsets), dtype=np.float)
        for edge_id in cut_edges + used_mtxs:
            cut_edge_imgs[edge_id] = 1
        cut_edge_imgs = cut_edge_imgs.reshape((len(self.mtx_offsets),)+self.img_shape)
        if self.use_bbox:
            ymax_vals, xmax_vals = self._bbox(cut_edge_imgs)
            self.bbox = [np.max(ymax_vals), np.max(xmax_vals)]
        else:
            self.bbox = cut_edge_imgs.shape
        self.state = np.concatenate((cut_edge_imgs[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]],
                                     self.current_affs[self.mtx_separating_channel:, 0:self.bbox[0], 0:self.bbox[1]]),
                                    axis=0).astype(np.float)
        self.counter += 1
        if self.counter == 20:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            labels = node_labeling.reshape(self.img_shape)
            complete_labeling = complete_labeling.reshape(self.img_shape)
            show_complete_seg = cm.prism(complete_labeling / complete_labeling.max())
            show_seg = cm.prism(labels / labels.max())
            img = np.concatenate([np.concatenate([cm.gray(self.current_affs[1]), show_complete_seg], axis=1),
                                  np.concatenate([cm.gray(self.current_affs[2]), cm.gray(self.current_affs[3])], axis=1)],
                                 axis=0)
            plt.imshow(img);plt.show()
            self.counter = 0

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

    def execute_action(self, actions):
        num_actions = 0
        print(actions)
        for idx, channel_actions in enumerate(actions):
            mask = np.array((channel_actions == 1) / 2 + (channel_actions == 2) * 2, dtype=np.float)
            mask_inv = np.array(mask == 0, dtype=np.float)
            num_actions += np.sum(mask != 0)
            self.current_affs[self.mtx_separating_channel+idx, 0:self.bbox[0], 0:self.bbox[1]] = \
                self.current_affs[self.mtx_separating_channel+idx, 0:self.bbox[0], 0:self.bbox[1]] * mask + \
                self.current_affs[self.mtx_separating_channel+idx, 0:self.bbox[0], 0:self.bbox[1]] * mask_inv
        self._update_state()
        reward = -num_actions/10 + 10 + np.sum(self.state[:self.mtx_separating_channel, 0:self.bbox[0], 0:self.bbox[1]].ravel()*2-1 ==
                                            self.gt_affinities[:, 0:self.bbox[0], 0:self.bbox[1]].ravel()) - 3 \
                 + np.finfo(np.float).eps
        self.accumulated_reward += reward
        print(f"reward: {reward}")
        if reward > self.quality or self.accumulated_reward < self.stop_reward:
            self.done = True
        return self.state, reward

    def reset(self):
        self.current_affs = self.initial_affs
        self.mtx_wtsd_max_iter = self.mtx_wtsd_start_iter
        self.done = False
        self.accumulated_reward = 0
