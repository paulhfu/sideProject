from models.ril_function_models import DNDQNRI,DNDQN
import torch
from torch.autograd import Variable
import numpy as np
from agents.qlagent import QlAgent
from utils import add_rndness_in_dis
import os

class RIAgentA2c(object):
    def __init__(self, gamma, n_state_channels, n_actions, action_shape, device, epsilon=0.4, writer=None):

        self.gamma = gamma
        self.eps = epsilon
        self.writer = writer
        self.action_shape = action_shape
        self.policy = DNDQNRI(num_classes=n_actions, num_inchannels=n_state_channels, device=device, block_config=(6, ))
        self.v_eval = DNDQN(num_classes=n_actions, num_inchannels=n_state_channels, device=device, block_config=(6, ))
        self.policy.cuda(device=self.policy.device)
        self.v_eval.cuda(device=self.v_eval.device)
        self.n_actions = n_actions
        self.sz_actions = np.prod(action_shape)
        self.n_state_channels = n_state_channels
        self.steps = 0

    def safe_model(self, directory):
        torch.save(self.policy.state_dict(), os.path.join(directory, 'mnm_a2c'))

    def load_model(self, directory):
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'mnm_a2c')), strict=True)

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
        log_probs = torch.log(action_probs.gather(-1, torch.from_numpy(action).to(self.policy.device)))
        value = values.gather(-1, torch.from_numpy(action).to(self.v_eval.device))
        action = action.item()
        return action, log_probs, value, entropy

    def learn(self, rewards, log_probs, values, policy_entropy, last_q_val):

        # compute Q values
        q_val = last_q_val.item()
        q_vals = np.zeros_like(values).tolist()
        for t in reversed(range(len(rewards))):
            q_val = rewards[t] + self.gamma * q_val
            q_vals[t] = q_val

        q_vals = torch.tensor(q_vals, dtype=torch.float32).to(self.policy.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.policy.device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantage = (q_vals - values).detach()
        advantage_error = (q_vals - values) - rewards
        policy_gradient_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage_error.pow(2).mean()
        loss = policy_gradient_loss + critic_loss + 0.001 * policy_entropy


        self.policy.optimizer.zero_grad()
        self.v_eval.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
        self.v_eval.optimizer.step()
        self.steps += 1
