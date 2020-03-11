# this agent implements the opposd algorithm introduced here : http://auai.org/uai2019/proceedings/papers/440.pdf
# using the trust region policy introduced by the acer algorithm: https://arxiv.org/pdf/1611.01224.pdf

from models.ril_function_models import UnetDQN, UnetRI
from agents.distribution_correction import DensityRatio
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeAngleConv1
from models.sp_embed_unet import SpVecsUnet
import torch
import numpy as np
import os
import torch.nn as nn
from agents.qlagent import QlAgent1
from utils.cstm_tensor import CstmTensor


class OPPOSDAgentUnet(QlAgent1):
    def __init__(self, gamma, lambdA, n_actions, device, env,
                 epsilon=0.9, eps_min=0.000001, replace_cnt=1, mem_size=10,  writer=None, train_phase=0):
        super(OPPOSDAgentUnet, self).__init__(gamma=gamma, eps=epsilon, eps_min=eps_min, replace_cnt=replace_cnt,
                                          mem_size=mem_size)

        self.train_phase = train_phase
        self.writer = writer
        self.env = env
        self.lambdA = lambdA
        self.q_eval = GcnEdgeAngleConv1(n_node_channels_in=1, n_edge_features_in=10, n_edge_classes=3, device=device, softmax=False)
        self.q_next = GcnEdgeAngleConv1(n_node_channels_in=1, n_edge_features_in=10, n_edge_classes=3, device=device, softmax=False)
        self.policy = GcnEdgeAngleConv1(n_node_channels_in=1, n_edge_features_in=10, n_edge_classes=3, device=device, softmax=True)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_next.device)
        self.policy.cuda(device=self.policy.device)
        self.n_actions = n_actions
        self.writer_idx1 = 0
        self.writer_idx2 = 0
        self.writer_idx3 = 0

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

    def learn(self):
        if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        c_loss, a_loss = 0, 0
        transition_data = self.mem.memory
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 0
        # self.dist_correction.update_density(transition_data, self.gamma)
        for t in transition_data:
            if not t.terminal:
                with torch.set_grad_enabled(False):
                    qvals_ = self.q_next(self.env.node_features.to(self.q_eval.device),
                                        self.env.edge_features.to(self.q_eval.device),
                                        self.env.edge_ids.to(self.q_eval.device),
                                        self.env.edge_angles.to(self.q_eval.device),
                                        t.state_.to(self.q_eval.device))
                    qvals_ = qvals_.squeeze().detach()
            with torch.set_grad_enabled(True):
                qvals = self.q_eval(self.env.node_features.to(self.q_eval.device),
                                     self.env.edge_features.to(self.q_eval.device),
                                     self.env.edge_ids.to(self.q_eval.device),
                                     self.env.edge_angles.to(self.q_eval.device),
                                     t.state.to(self.q_eval.device))
                qvals = qvals.squeeze()
            pvals = nn.functional.softmax(qvals, -1).detach()
            actions = t.actions.to(self.q_eval.device)

            m = m + self.gamma ** (t.time-current) * importance_weight
            importance_weight = importance_weight * self.lambdA * \
                                torch.min(torch.ones(actions.unsqueeze(-1).shape).to(self.q_eval.device),
                                          pvals.gather(-1, actions.unsqueeze(-1)) / t.behav_probs.to(self.q_eval.device)).squeeze()
            if t.terminal:
                c_loss = c_loss + self.q_eval.loss(t.reward.to(self.q_eval.device) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)
            else:
                c_loss = c_loss + self.q_eval.loss((t.reward.to(self.q_eval.device) + self.gamma * qvals_.max(-1)[0]) * m, qvals.gather(-1, actions.unsqueeze(-1)).squeeze() * m)

        c_loss = c_loss / len(transition_data)
        self.q_eval.optimizer.zero_grad()
        c_loss.backward()
        self.q_eval.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar("loss/critic", c_loss.item(), self.writer_idx1)
            self.writer_idx1 += 1

        # sample according to discounted state dis
        discount_distribution = [self.gamma**i for i in range(len(transition_data))]
        discount_distribution = np.exp(discount_distribution) / sum(np.exp(discount_distribution))  # softmax
        batch_ind = np.random.choice(len(transition_data), size=len(transition_data), p=discount_distribution)
        z = 0
        for i in batch_ind:
            t = transition_data[i]
            # w = self.dist_correction.density_ratio(t.state.unsqueeze(0).unsqueeze(0).to(self.dist_correction.density_ratio.device)).detach().squeeze()
            w = 1
            z += w
            with torch.set_grad_enabled(False):
                qvals = self.q_eval(self.env.node_features.to(self.q_eval.device),
                                    self.env.edge_features.to(self.q_eval.device),
                                    self.env.edge_ids.to(self.q_eval.device),
                                    self.env.edge_angles.to(self.q_eval.device),
                                    t.state.to(self.q_eval.device)).squeeze().to(self.policy.device)
            with torch.set_grad_enabled(True):
                policy_values = self.policy(self.env.node_features.to(self.policy.device),
                                            self.env.edge_features.to(self.policy.device),
                                            self.env.edge_ids.to(self.policy.device),
                                            self.env.edge_angles.to(self.policy.device),
                                            t.state.to(self.policy.device)).squeeze()
            action_probs = policy_values.gather(-1, t.actions.unsqueeze(-1).to(self.policy.device)).squeeze()
            q_vals = qvals.gather(-1, t.actions.to(self.q_eval.device).unsqueeze(-1)).squeeze()
            v_vals = qvals * policy_values.detach()
            v_vals = v_vals.sum(-1)
            advantage = q_vals - v_vals
            a_loss = a_loss - (action_probs.detach()/t.behav_probs.to(self.policy.device)) * w * torch.log(action_probs) * advantage
        z = z / len(batch_ind)
        a_loss = a_loss / z
        a_loss = a_loss / len(batch_ind)
        a_loss = torch.sum(a_loss)
        a_loss = a_loss / len(t.state)

        self.policy.optimizer.zero_grad()
        a_loss.backward()
        self.policy.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar("loss/actor", a_loss.item(), self.writer_idx2)
            self.writer_idx2 += 1

        self.learn_steps += 1
        return

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'c_opposd_ri_gcn'))
        torch.save(self.policy.state_dict(), os.path.join(directory, 'a_opposd_ri_gcn'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'c_opposd_ri_gcn')), strict=True)
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'a_opposd_ri_gcn')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state, count):
        with torch.set_grad_enabled(False):
            action_probs = self.policy(self.env.node_features.to(self.policy.device),
                                     self.env.edge_features.to(self.policy.device),
                                     self.env.edge_ids.to(self.policy.device),
                                     self.env.edge_angles.to(self.policy.device),
                                     state.to(self.policy.device))
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

