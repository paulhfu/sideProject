from models.GCNNs.mc_glbl_edge_costs import GcnEdgeConvNet4
from agents.qlagent import QlAgent1
from matplotlib import cm
import matplotlib.pyplot as plt
from agents.replayMemory import Transition
import numpy as np
import torch.nn as nn
import torch
import os


class QlAgentGcn1(QlAgent1):
    """Agent for Q learning using DenseNet for Q-func"""

    def __init__(self, gamma, lambdA, n_actions, device, env,
                 epsilon=0.9, eps_min=0.000001, replace_cnt=5, mem_size=10,  writer=None):
        super(QlAgentGcn1, self).__init__(gamma=gamma, eps=epsilon, eps_min=eps_min, replace_cnt=replace_cnt, mem_size=mem_size)

        self.writer = writer
        self.env = env
        self.lambdA = lambdA
        self.q_eval = GcnEdgeConvNet4(n_node_features_in=3, n_edge_features_in=1, n_edge_classes=3, device=device, softmax=False)
        self.q_next = GcnEdgeConvNet4(n_node_features_in=3, n_edge_features_in=1, n_edge_classes=3, device=device, softmax=False)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = n_actions

    def get_qLoss(self):
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        # Focus on first on the q values being correct
        loss = 0
        transition_data = self.mem.memory
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 0
        for t in transition_data:
            with torch.set_grad_enabled(False):
                qvals_ = self.q_next(self.env.node_features.to(self.q_eval.device),
                                     t.state_.to(self.q_eval.device),
                                     self.env.edge_ids.to(self.q_eval.device)).squeeze().detach()
            with torch.set_grad_enabled(True):
                qvals = self.q_eval(self.env.node_features.to(self.q_eval.device),
                                     t.state.to(self.q_eval.device),
                                     self.env.edge_ids.to(self.q_eval.device)).squeeze()
            pvals_ = nn.functional.softmax(qvals_, -1).detach()
            pvals = nn.functional.softmax(qvals, -1).detach()
            actions = t.action.to(self.q_eval.device)

            m = m + self.gamma ** (t.time-current) * importance_weight
            importance_weight = importance_weight * self.lambdA * \
                                torch.min(torch.ones(actions.unsqueeze(-1).shape).to(self.q_eval.device),
                                          pvals.gather(-1, actions.unsqueeze(-1)) / t.behav_probs.to(self.q_eval.device)).squeeze()
            loss = loss + self.q_eval.loss((t.reward.to(self.q_eval.device) + self.gamma*torch.sum(qvals_ * pvals_, -1)) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)
        loss = loss / len(transition_data)
        return loss.mean()

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'ql_retrace_ri'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'ql_retrace_ri')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state):
        action_probs = nn.functional.softmax(self.q_eval(self.env.node_features.to(self.q_eval.device),
                                                         state.to(self.q_eval.device),
                                                         self.env.edge_ids.to(self.q_eval.device)).squeeze().detach(), -1)
        behav_probs = action_probs + self.eps * (1/self.n_actions - action_probs)
        actions = torch.multinomial(behav_probs, 1).squeeze()
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
