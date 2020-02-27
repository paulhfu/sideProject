# this agent implements the opposd algorithm introduced here : http://auai.org/uai2019/proceedings/papers/440.pdf
# using the trust region policy introduced by the acer algorithm: https://arxiv.org/pdf/1611.01224.pdf

from models.ril_function_models import UnetDQN, UnetRI
from agents.distribution_correction import DensityRatio
import torch
import numpy as np
import os
import torch.nn as nn


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

    def get_loss(self):
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        c_loss, a_loss = 0, 0
        transition_data = self.mem.memory
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 0
        z = 0
        self.dist_correction.update_density(transition_data)
        for t in transition_data:
            if not t.terminal:
                with torch.set_grad_enabled(False):
                    qvals_, tt = self.q_next(self.env.node_features.to(self.q_eval.device),
                                         torch.cat((t.state_, self.env.initial_edge_weights), -1).to(self.q_eval.device),
                                         self.env.edge_ids.to(self.q_eval.device))
                    qvals_ = qvals_.squeeze().detach()
                    assert all(tt.cpu().squeeze().detach() == self.env.gt_edge_weights.squeeze())
                    pvals_ = nn.functional.softmax(qvals_, -1).detach()
            with torch.set_grad_enabled(True):
                qvals, tt = self.q_eval(self.env.node_features.to(self.q_eval.device),
                                     torch.cat((t.state, self.env.initial_edge_weights), -1).to(self.q_next.device),
                                     self.env.edge_ids.to(self.q_eval.device))
                qvals = qvals.squeeze()
                assert all(tt.cpu().squeeze().detach() == self.env.gt_edge_weights.squeeze())
            pvals = nn.functional.softmax(qvals, -1).detach()
            actions = t.action.to(self.q_eval.device)

            m = m + self.gamma ** (t.time-current) * importance_weight
            importance_weight = importance_weight * self.lambdA * \
                                torch.min(torch.ones(actions.unsqueeze(-1).shape).to(self.q_eval.device),
                                          pvals.gather(-1, actions.unsqueeze(-1)) / t.behav_probs.to(self.q_eval.device)).squeeze()
            if t.terminal:
                c_loss += c_loss + self.q_eval.loss(t.reward.to(self.q_eval.device) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)
            else:
                c_loss += c_loss + self.q_eval.loss((t.reward.to(self.q_eval.device) + self.gamma * qvals_.max(-1)[0]) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)

        a_loss = a_loss / len(transition_data)

        # sample according to discounted state dis
        discount_distribution = [self.gamma ** (i.time + 1) for i in transition_data]
        batch = np.random.choice(transition_data, size=len(transition_data), p=discount_distribution)
        for t in batch:
            z += self.dist_correction.density_ratio(t.state.to(self.dist_correction.density_ratio.device))
        for t in batch:
            w = self.dist_correction.density_ratio(t.state.to(self.dist_correction.density_ratio.device))
            a_loss +=  t.policy_ratio * w / z * torch.log(t.behav_probs) * self.q_eval()
        a_loss = a_loss / len(batch)
        return c_loss, a_loss

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'ql_retrace_ri_gcn'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'ql_retrace_ri_gcn')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state, count):
        action_probs, tt = self.policy(self.env.node_features.to(self.q_eval.device),
                                 torch.cat((state, self.env.initial_edge_weights), -1).to(self.q_eval.device),
                                 self.env.edge_ids.to(self.q_eval.device))
        assert all(tt.cpu().squeeze().detach() == self.env.gt_edge_weights.squeeze())
        if self.train_phase == 0:
            behav_probs = action_probs + self.eps * (1 / self.n_actions - action_probs)
            actions = torch.multinomial(behav_probs, 1).squeeze()
            sel_behav_probs = behav_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
        elif self.train_phase == 1:
            randm_draws = int(self.eps * len(action_probs))
            if randm_draws > 0:
                actions = action_probs.max(-1)[1].squeeze()
                if self.writer is not None and count == 0 and self.learn_steps%5 == 0:
                    self.writer.add_text("actions", np.array2string(actions.cpu().numpy(), precision=2, separator=',', suppress_small=True))
                randm_indices = torch.multinomial(torch.ones(len(action_probs))/len(action_probs), randm_draws)
                actions[randm_indices] = torch.randint(0, self.n_actions, (randm_draws, )).to(self.q_eval.device)
                sel_behav_probs = action_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
                sel_behav_probs[randm_indices] = 1 / self.n_actions
            else:
                actions = action_probs.max(-1)[1].squeeze()
                if self.writer is not None and count == 0 and self.learn_steps%5 == 0:
                    self.writer.add_text("actions", np.array2string(actions.cpu().numpy(), precision=2, separator=',', suppress_small=True))
                sel_behav_probs = action_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
        else:
            actions = action_probs.max(-1)[1].squeeze()
            sel_behav_probs = action_probs.gather(-1, actions.unsqueeze(-1)).squeeze()

        # log_probs = torch.log(sel_behav_probs)
        # entropy = - (behav_probs * torch.log(behav_probs)).sum()
        return actions.cpu(), sel_behav_probs.cpu()

    def learn(self):
        if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        qloss, p_loss = self.get_loss()
        if self.writer is not None:
            self.writer.add_scalar("loss", qloss.item())
        self.q_eval.optimizer.zero_grad()
        qloss.backward()
        # for param in self.q_eval.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-1, 1)
        self.q_eval.optimizer.step()
        self.learn_steps += 1