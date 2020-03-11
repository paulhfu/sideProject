# this file implements the solutions presented by this publication: https://arxiv.org/pdf/1810.12429.pdf
import numpy as np
from agents.replayMemory import Transition_t
from models.conv_net_1d import UNet1d
import torch

class DensityRatio(object):

    def __init__(self, n_state_channels, device, agent, kernel='gauss', gauss_sigma=None, poly_offs=None, poly_degree=None, lapl_alpha=None, writer=None):
        self.agent = agent
        self.writer = writer
        self.writer_idx = 0
        self.gauss_sigma = gauss_sigma
        self.poly_offs = poly_offs
        self.poly_degree = poly_degree
        self.lapl_alpha = lapl_alpha
        self.density_ratio = UNet1d(n_state_channels, 1, device=device)
        self.density_ratio.cuda(device=self.density_ratio.device)
        if kernel == 'polynomial':
            self.kernel_function = lambda obj, x, y: (np.dot(x, y) + obj.poly_offs) ** obj.poly_degree
        elif kernel == 'linear':
            self.kernel_function = lambda obj, x, y: np.dot(x, y)
        elif kernel == 'laplacian':
            self.kernel_function = lambda obj, x, y: np.exp(-obj.lapl_alpha(np.linalg.norm(x-y)))
        else:  # Gauss kernel
            self.kernel_function = lambda obj, x, y: torch.exp(-((x-y) ** 2) / (2 * obj.gauss_sigma))

    def warm_start(self, targets, states):
        for state, target in zip(states, targets):
            loss = self.density_ratio.loss(self.density_ratio(state), target)
            self.density_ratio.optimizer.zero_grad()
            loss.backward()
            for param in self.density_ratio.parameters():
                param.grad.data.clamp_(-1, 1)
            self.density_ratio.optimizer.step()

    def _transition_delta(self, transition, action_probs):
        with torch.set_grad_enabled(True):
            if transition.time == -1:
                ret = 1-self.density_ratio(transition.state_.unsqueeze(0).unsqueeze(0).to(self.density_ratio.device))
                return ret

            ret = (self.density_ratio(transition.state.unsqueeze(0).unsqueeze(0).to(self.density_ratio.device))
                   * (action_probs / transition.behav_probs.to(self.density_ratio.device)))\
                  - self.density_ratio(transition.state_.unsqueeze(0).unsqueeze(0).to(self.density_ratio.device))
            return ret

    def update_density(self, transit_data, gamma=1, batch_size=10, steps=1):
        transit_data.insert(0, Transition_t(transit_data[0].state, 0, 0, transit_data[0].state, -1, 0, False))
        discount_distribution = [gamma**i for i in range(len(transit_data))]
        discount_distribution = np.exp(discount_distribution) / sum(np.exp(discount_distribution))  # softmax
        for step in range(steps):
            action_probs = self._get_all_action_probs(transit_data)
            batch_ind = np.random.choice(len(transit_data), size=batch_size, p=discount_distribution)
            cnt = 0
            norm_const = 0
            loss = 0
            with torch.set_grad_enabled(False):
                for i in batch_ind:
                    norm_const += self.density_ratio(transit_data[i].state.unsqueeze(0).unsqueeze(0).to(self.density_ratio.device))
            norm_const /= len(batch_ind)
            interm_loss = 0
            for i in batch_ind:
                # for j in batch_ind:
                #     if j == i:
                #         continue
                if len(batch_ind[batch_ind != i]) == 0:
                    continue
                j = np.random.choice(batch_ind[batch_ind != i])
                cnt += 1
                transition_delta_1 = self._transition_delta(transit_data[i], action_probs[i]) / norm_const
                if any(torch.isnan(transition_delta_1).view(-1)) or any(torch.isinf(transition_delta_1).view(-1)):
                    a=1
                transition_delta_2 = self._transition_delta(transit_data[j], action_probs[j]) / norm_const
                if any(torch.isnan(transition_delta_2).view(-1)) or any(torch.isinf(transition_delta_2).view(-1)):
                    a=1
                interm_loss = -((transition_delta_1 * transition_delta_2 * self.kernel_function(self, transit_data[i].state_, transit_data[j].state_).to(self.density_ratio.device))) ** 2 / len(batch_ind)
                loss = loss + torch.sum(interm_loss) / len(transit_data[0].state)
            if cnt > 0:
                # loss = loss + torch.sum(interm_loss)
                # loss = loss/len(transit_data[0].state)
                self.density_ratio.optimizer.zero_grad()
                loss.backward()
                # for param in self.density_ratio.parameters():
                #     param.grad.data.clamp_(-1, 1)
                self.density_ratio.optimizer.step()

                if self.writer is not None:
                    self.writer.add_scalar("loss/density_ratio", loss.item(), self.writer_idx)
                    self.writer_idx += 1
        del transit_data[0]
        return None


    def _get_all_action_probs(self, transitions):
        ret = []
        with torch.set_grad_enabled(False):
            for transition in transitions:
                if transition.time == -1:
                    ret.append(0)
                else:
                    action_probs = self.agent.policy(self.agent.env.node_features.to(self.agent.policy.device),
                                                     self.agent.env.edge_features.to(self.agent.policy.device),
                                                     self.agent.env.edge_ids.to(self.agent.policy.device),
                                                     self.agent.env.edge_angles.to(self.agent.policy.device),
                                                     transition.state.to(self.agent.policy.device)).to(
                        self.density_ratio.device)
                    ret.append(action_probs.gather(-1, transition.actions.unsqueeze(-1).to(self.density_ratio.device)).squeeze())
        return ret