# this file implements the solutions presented by this publication: https://arxiv.org/pdf/1810.12429.pdf
import numpy as np
from agents.replayMemory import Transition_t
from models.ril_function_models import DNDQN
import torch

class DensityRatio(object):

    def __init__(self, n_state_channels, device, tgt_policy, b_policy, kernel='gauss', gauss_sigma=None, poly_offs=None, poly_degree=None, lapl_alpha=None):
        self.tgt_policy = tgt_policy
        self.b_policy = b_policy
        self.gauss_sigma = gauss_sigma
        self.poly_offs = poly_offs
        self.poly_degree = poly_degree
        self.lapl_alpha = lapl_alpha
        self.density_ratio = DNDQN(num_classes=1, num_inchannels=n_state_channels, device=device)
        self.density_ratio.cuda(device=self.q_eval.device)
        if kernel == 'polynomial':
            self.kernel_function = lambda obj, x, y: (np.dot(x, y) + obj.poly_offs) ** obj.poly_degree
        elif kernel == 'linear':
            self.kernel_function = lambda obj, x, y: np.dot(x, y)
        elif kernel == 'laplacian':
            self.kernel_function = lambda obj, x, y: np.exp(-obj.lapl_alpha(np.linalg.norm(x-y)))
        else:  # Gauss kernel
            self.kernel_function = lambda obj, x, y: np.exp(-(np.linalg.norm(x-y) ** 2) / (2 * obj.gauss_sigma))

    def warm_start(self, targets, states):
        for state, target in zip(states, targets):
            loss = self.density_ratio.loss(self.density_ratio(state), target)
            self.density_ratio.optimizer.zero_grad()
            loss.backward()
            for param in self.density_ratio.parameters():
                param.grad.data.clamp_(-1, 1)
            self.density_ratio.optimizer.step()

    def _transition_delta(self, transition, norm_const):
        if transition.time == -1:
            return 1-self.density_ratio(transition.next_state)
        else:
            return (self.density_ratio(transition.state) / norm_const) * (self.tgt_policy(transition.state)[transition.action] / transition.behavior_proba) - (self.density_ratio(transition.state_) / norm_const)

    def update_density(self, transit_data, gamma=1, batch_size=10, steps=10):
        transit_data.push(0, 0, transit_data.memory[0].next_state, -1, 0, False)
        discount_distribution = [gamma**(i.time+1) for i in transit_data.memory]
        discount_distribution = np.exp(discount_distribution) / sum(np.exp(discount_distribution))  # softmax
        for step in steps:
            batch = Transition_t(*zip(*np.random.choice(transit_data.memory, size=batch_size, p=discount_distribution)))
            bin_coeff = ((len(batch) * (len(batch) + 1)) / 2) + len(batch)
            cnt = 0
            norm_const = 0
            loss = 0
            for i in range(len(batch)):
                for j in range(i, len(batch)):
                    norm_const += self.density_ratio(batch[i].state)
            norm_const /= (1/bin_coeff)
            for i in range(len(batch)):
                for j in range(i, len(batch)):
                    cnt += 1
                    transition_delta_1 = self._transition_delta(batch[i], norm_const)
                    transition_delta_2 = self._transition_delta(batch[j], norm_const)
                    loss += transition_delta_1*transition_delta_2*self.kernel_function(batch[i].state_, batch[j].state_)
            loss /= bin_coeff
            self.density_ratio.optimizer.zero_grad()
            loss.backward()
            for param in self.density_ratio.parameters():
                param.grad.data.clamp_(-1, 1)
            self.density_ratio.optimizer.step()
        return None
