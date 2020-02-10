# this agent implements the opposd algorithm introduced here : http://auai.org/uai2019/proceedings/papers/440.pdf
# using the trust region policy introduced by the acer algorithm: https://arxiv.org/pdf/1611.01224.pdf

from models.ril_function_models import UnetDQN, UnetFcnDQN, UnetRI
from distribution_correction import DensityRatio
from agents.replayMemory import Transition_t
import torch
from torch.autograd import Variable
import numpy as np
from agents.qlagent import QlAgent
from utils import add_rndness_in_dis
import os


class OPPOSDAgentUnet(object):
    def __init__(self, gamma, lambdA , n_state_channels, n_actions, action_shape, device, img_shape, epsilon=0.4, writer=None):

        self.gamma = gamma
        self.lambdA = lambdA
        self.eps = epsilon
        self.writer = writer
        self.action_shape = action_shape
        self.policy = UnetRI(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=len(action_shape), device=device)
        self.q_val = UnetDQN(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=n_actions, device=device)
        self.policy.cuda(device=self.policy.device)
        self.n_actions = n_actions
        self.sz_actions = np.prod(action_shape)
        self.n_state_channels = n_state_channels
        self.dist_correction = DensityRatio(self, n_state_channels, device, self.policy, kernel='gauss', gauss_sigma=.3)
        self.dist_correction.warm_start(torch.ones(40), torch.randn(40, n_state_channels, img_shape[0], img_shape[1]))
        self.steps = 0

    def expected_qval(self, qvals, pvals):
        # this corresponds to V
        e = 0
        for a, p in zip(qvals, pvals):
            e += a*p
        return e

    def warm_start(self, transition_data):
        # warm-starting with data from initial behavior policy which is assumed to be uniform distribution
        qloss = self.get_qLoss(transition_data)
        ploss = 0
        for t in transition_data:
            ploss += self.policy.loss(self.policy(t.state), torch.ones(self.action_shape)/2)
        ploss /= len(transition_data)
        self.q_val.optimizer.zero_grad()
        qloss.backward()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_val.optimizer.step()
        self.policy.optimizer.zero_grad()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()

    def get_qLoss(self, transition_data):
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        loss = 0
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 1
        for t, t_ in reversed(zip(transition_data[:-1], transition_data[1:])):
            qvals_ = self.q_val(t_.state)
            pvals_ = self.policy(t_.state)
            qvals = self.q_val(t.state)

            m += self.gamma ** (t.time-current) * importance_weight
            importance_weight *= self.lambdA * min(1, self.policy(transition_data[t.time].state)[transition_data[t.time].action].detach().cpu().numpy() / transition_data[t.time].behavior_proba)

            loss += t.reward + self.gamma*self.expected_qval(qvals_, pvals_) - qvals * m
        loss /= len(transition_data)
        return loss

    def safe_model(self, directory):
        torch.save(self.policy.state_dict(), os.path.join(directory, 'mnm_ri'))

    def load_model(self, directory):
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'mnm_ri')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state, by_max=False):
        action_probs = self.policy(torch.tensor(state, dtype=torch.float32, requires_grad=True).to(self.policy.device).unsqueeze(0)).squeeze()
        values = self.v_eval(torch.tensor(state, dtype=torch.float32, requires_grad=True).to(self.v_eval.device).unsqueeze(0)).squeeze()
        if by_max:
            action = np.array(action_probs.detach().cpu().numpy().argmax(axis=-1))
        else:
            rnd_pol_dis = add_rndness_in_dis(np.squeeze(action_probs.detach().cpu().numpy()).reshape(-1, self.n_actions)
                                             , self.eps)
            action = np.array([np.random.choice(self.n_actions, p=dis) for dis in rnd_pol_dis])
        entropy = - (action_probs * torch.log(action_probs)).sum()
        action = action.item()
        return action, entropy

    def learn_a2c(self, trajectories, batch_size=5, steps=5):
        # update state distribution approximation d_pi(s)/d_mu(s)
        self.dist_correction.update_density(transit_data=trajectories, gamma=1, batch_size=batch_size)
        # update the criric
        qloss = self.get_qLoss(trajectories.memory)
        self.q_val.optimizer.zero_grad()
        qloss.backward()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_val.optimizer.step()
        # update the actor
        discount_distribution = [self.gamma**(i.time+1) for i in trajectories.memory]
        discount_distribution = np.exp(discount_distribution) / sum(np.exp(discount_distribution))
        for step in steps:
            batch = Transition_t(*zip(*np.random.choice(trajectories.memory, size=batch_size, p=discount_distribution)))
            dist_correction = self.dist_correction.density_ratio(batch.state)
            norm_coeff = dist_correction.sum() / batch_size
            action_probs = self.policy(
                torch.tensor(batch.state, dtype=torch.float32, requires_grad=True).to(self.policy.device).unsqueeze(
                    0)).squeeze()
            probs = self.action_probs.gather(-1, torch.from_numpy(batch.action).to(self.policy.device))
            log_probs = torch.log(probs)
            proba_mismatch = probs / batch.behavior_proba
            ploss = torch.sum(dist_correction / norm_coeff * proba_mismatch * log_probs * self.q_val(batch.state)) / batch_size
            self.policy.optimizer.zero_grad()
            ploss.backward()
            # for param in self.q_eval.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.policy.optimizer.step()

        self.q_val.optimizer.zero_grad()
        qloss.backward()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_val.optimizer.step()
        self.steps += 1