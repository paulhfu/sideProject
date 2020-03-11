from models.GCNNs.mc_glbl_edge_costs import GcnEdgeConvNet4, GcnEdgeConvNet3, GcnEdgeAngleConv1
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
                 epsilon=0.9, eps_min=0.000001, replace_cnt=1, mem_size=10,  writer=None, train_phase=0):
        super(QlAgentGcn1, self).__init__(gamma=gamma, eps=epsilon, eps_min=eps_min, replace_cnt=replace_cnt, mem_size=mem_size)

        self.train_phase = train_phase
        self.writer = writer
        self.env = env
        self.lambdA = lambdA
        # self.q_eval = GcnEdgeConvNet4(n_node_features_in=1, n_edge_features_in=2, n_edge_classes=3, device=device, softmax=False)
        # self.q_next = GcnEdgeConvNet4(n_node_features_in=1, n_edge_features_in=2, n_edge_classes=3, device=device, softmax=False)
        self.q_eval = GcnEdgeAngleConv1(n_node_channels_in=1, n_edge_features_in=10, n_edge_classes=3, device=device, softmax=False)
        self.q_next = GcnEdgeAngleConv1(n_node_channels_in=1, n_edge_features_in=10, n_edge_classes=3, device=device, softmax=False)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = n_actions
        self.writer_idx1 = 0
        self.writer_idx2 = 0
        self.writer_idx3 = 0

    def get_qLoss(self):
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        loss = 0
        transition_data = self.mem.memory
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 0
        for t in transition_data:
            if not t.terminal:
                with torch.set_grad_enabled(False):
                    qvals_ = self.q_next(self.env.node_features.to(self.q_eval.device),
                                        self.env.edge_features.to(self.q_eval.device),
                                        self.env.edge_ids.to(self.q_eval.device),
                                        self.env.edge_angles.to(self.q_eval.device),
                                        t.state_.to(self.q_eval.device))
                    qvals_ = qvals_.squeeze().detach()
                    # assert all(tt.cpu().squeeze().detach() == self.env.gt_edge_weights.squeeze())
                    # pvals_ = nn.functional.softmax(qvals_, -1).detach()
            with torch.set_grad_enabled(True):
                qvals = self.q_eval(self.env.node_features.to(self.q_eval.device),
                                     self.env.edge_features.to(self.q_eval.device),
                                     self.env.edge_ids.to(self.q_eval.device),
                                     self.env.edge_angles.to(self.q_eval.device),
                                     t.state.to(self.q_eval.device))
                qvals = qvals.squeeze()
                # assert all(tt.cpu().squeeze().detach() == self.env.gt_edge_weights.squeeze())
            pvals = nn.functional.softmax(qvals, -1).detach()
            actions = t.action.to(self.q_eval.device)

            m = m + self.gamma ** (t.time-current) * importance_weight
            importance_weight = importance_weight * self.lambdA * \
                                torch.min(torch.ones(actions.unsqueeze(-1).shape).to(self.q_eval.device),
                                          pvals.gather(-1, actions.unsqueeze(-1)) / t.behav_probs.to(self.q_eval.device)).squeeze()
            # m = 1
            if t.terminal:
                loss = loss + self.q_eval.loss(t.reward.to(self.q_eval.device) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)
            else:
                loss = loss + self.q_eval.loss((t.reward.to(self.q_eval.device) + self.gamma * qvals_.max(-1)[0]) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)
        # loss = loss / len(transition_dataedge_features_1d)
        return loss

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'ql_retrace_ri_gcn'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'ql_retrace_ri_gcn')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state, count):
        # q_vals, tt = self.q_eval(self.env.node_features.to(self.q_eval.device),
        #                          torch.cat((state, self.env.initial_edge_weights), -1).to(self.q_eval.device),
        #                          self.env.edge_ids.to(self.q_eval.device))
        q_vals = self.q_eval(self.env.node_features.to(self.q_eval.device),
                                 self.env.edge_features.to(self.q_eval.device),
                                 self.env.edge_ids.to(self.q_eval.device),
                                 self.env.edge_angles.to(self.q_eval.device),
                                 state.to(self.q_eval.device))

        action_probs = nn.functional.softmax(q_vals.squeeze().detach(), -1)
        if self.train_phase == 0:
            behav_probs = action_probs + self.eps * (1 / self.n_actions - action_probs)
            actions = torch.multinomial(behav_probs, 1).squeeze()
            sel_behav_probs = behav_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
        elif self.train_phase == 1:
            randm_draws = int(self.eps * len(action_probs))
            if randm_draws > 0:
                actions = action_probs.max(-1)[1].squeeze()
                if self.writer is not None and count == 0 and self.learn_steps%5 == 0:
                    self.writer.add_text("actions", np.array2string(actions.cpu().numpy(), precision=2, separator=',',suppress_small=True), self.writer_idx1)
                    self.writer_idx1 += 1
                randm_indices = torch.multinomial(torch.ones(len(action_probs))/len(action_probs), randm_draws)
                actions[randm_indices] = torch.randint(0, self.n_actions, (randm_draws, )).to(self.q_eval.device)
                sel_behav_probs = action_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
                sel_behav_probs[randm_indices] = 1 / self.n_actions
            else:
                actions = action_probs.max(-1)[1].squeeze()
                if self.writer is not None and count == 0 and self.learn_steps%5 == 0:
                    self.writer.add_text("actions", np.array2string(actions.cpu().numpy(), precision=2, separator=',',suppress_small=True), self.writer_idx1)
                    self.writer_idx1 += 1
                sel_behav_probs = action_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
        else:
            actions = action_probs.max(-1)[1].squeeze()
            sel_behav_probs = action_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
        # entropy = - (behav_probs * torch.log(behav_probs)).sum()
        return actions.cpu(), sel_behav_probs.cpu()

    def learn(self):
        if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        qloss = self.get_qLoss()
        if self.writer is not None:
            self.writer.add_scalar("loss", qloss.item(), self.writer_idx2)
            self.writer_idx2 += 1
        self.q_eval.optimizer.zero_grad()
        qloss.backward()
        # for param in self.q_eval.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-1, 1)
        self.q_eval.optimizer.step()
        self.learn_steps += 1
