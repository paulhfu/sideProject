# this agent implements the opposd algorithm introduced here : http://auai.org/uai2019/proceedings/papers/440.pdf
# using the trust region policy introduced by the acer algorithm: https://arxiv.org/pdf/1611.01224.pdf

from models.ril_function_models import UnetDQN, UnetFcnDQN, UnetRI
from distribution_correction import DensityRatio
from agents.replayMemory import Transition_t
import torch
import torch.nn as nn
from agents.qlagent import QlAgent1
from torch.autograd import Variable
import numpy as np
from agents.qlagent import QlAgent
from utils import add_rndness_in_dis
import os


class QlRetraceAgentUnet(QlAgent1):
    def __init__(self, gamma, lambdA, n_actions, n_edges, device, env, epsilon=0.4, writer=None):
        super(QlRetraceAgentUnet, self).__init__(gamma=gamma, eps=epsilon, eps_min=0.000001, replace_cnt=4, mem_size=20)
        self.gamma = gamma
        self.lambdA = lambdA
        self.eps = epsilon
        self.writer = writer
        self.q_eval = UnetDQN(n_inChannels=n_edges, n_edges=n_edges, n_actions=n_actions, device=device)
        self.q_next = UnetDQN(n_inChannels=n_edges, n_edges=n_edges, n_actions=n_actions, device=device)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_next.device)
        self.n_actions = n_actions
        self.n_edges = n_edges
        self.env = env
        self.steps = 0

    def get_qLoss(self):
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        loss = 0
        transition_data = self.mem.memory
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 0
        for t in transition_data:
            qvals_ = self.q_next(torch.tensor(t.state_, dtype=torch.float32, requires_grad=False).to(self.q_eval.device).unsqueeze(0)).squeeze().detach()
            qvals = self.q_eval(torch.tensor(t.state, dtype=torch.float32, requires_grad=True).to(self.q_eval.device).unsqueeze(0)).squeeze()
            pvals_ = nn.functional.softmax(qvals_, -1).detach()
            pvals = nn.functional.softmax(qvals, -1).detach()
            actions = t.action.to(self.q_eval.device)

            m = m + self.gamma ** (t.time-current) * importance_weight
            importance_weight = importance_weight * self.lambdA * torch.stack([torch.ones(actions.unsqueeze(-1).shape).to(self.q_eval.device), pvals.gather(-1, actions.unsqueeze(-1)) / t.behav_probs.to(self.q_eval.device)], -1).min(-1)[0].squeeze()

            loss = loss + (t.reward.to(self.q_eval.device) + self.gamma*torch.sum(qvals_ * pvals_, -1) - qvals.gather(-1, actions.unsqueeze(-1)).squeeze()) ** 2 * m
        loss = loss / len(transition_data)
        return loss.mean()

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'ql_retrace_ri'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'ql_retrace_ri')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state):
        action_probs = nn.functional.softmax(self.q_eval(torch.tensor(state, dtype=torch.float32, requires_grad=False).to(self.q_eval.device).unsqueeze(0)).squeeze().detach(), -1)
        mask1 = (action_probs <= self.eps).float()
        mask2 = (action_probs > self.eps).float()
        masked_probs1 = mask1 * action_probs
        masked_probs2 = mask2 * action_probs
        diff = masked_probs1.max() - masked_probs2.max()
        behav_probs = (masked_probs1 + (diff * self.eps)) + (masked_probs2 - (diff * self.eps))

        actions = torch.multinomial(behav_probs.view(-1, self.n_actions), 1).view(behav_probs.shape[:-1])
        sel_behav_probs = behav_probs.gather(-1, actions.unsqueeze(-1))
        # entropy = - (behav_probs * torch.log(behav_probs)).sum()
        return actions.cpu(), sel_behav_probs.cpu()

    def learn(self):
        if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        qloss = self.get_qLoss()
        self.q_eval.optimizer.zero_grad()
        qloss.backward()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_eval.optimizer.step()
        self.learn_steps += 1

