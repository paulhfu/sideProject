from models.ril_function_models import DNDQN, UnetFcnDQN, UnetRI
import torch
from torch.autograd import Variable
import numpy as np
from agents.qlagent import QlAgent
from utils import add_rndness_in_dis
import os

class RIAgentUnet(object):
    def __init__(self, gamma, n_state_channels, n_edges, n_actions, action_shape, device, epsilon=0.9, writer=None):

        self.gamma = gamma
        self.eps = epsilon
        self.writer = writer
        self.action_shape = action_shape
        self.policy = UnetRI(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=len(action_shape), device=device)
        self.policy.cuda(device=self.policy.device)
        self.n_actions = n_actions
        self.sz_actions = np.prod(action_shape)
        self.n_state_channels = n_state_channels
        self.steps = 0

    def safe_model(self, directory):
        torch.save(self.policy.state_dict(), os.path.join(directory, 'unet_ri'))

    def load_model(self, directory):
        self.policy.load_state_dict(torch.load(os.path.join(directory, 'unet_ri')), strict=True)

    def reset_eps(self, eps):
        self.eps = eps

    def get_action(self, state, by_max=False):
        action_probs = self.policy(torch.tensor(state, dtype=torch.float32, requires_grad=True).to(self.policy.device).unsqueeze(0)).squeeze()
        if by_max:
            action = action_probs.detach().cpu().numpy().argmax(axis=-1)
        else:
            rnd_pol_dis = add_rndness_in_dis(np.squeeze(action_probs.detach().cpu().numpy()).reshape(-1, self.n_actions), self.eps)
            action = np.array([np.random.choice(self.n_actions, p=dis) for dis in rnd_pol_dis])
            action = action.reshape(self.action_shape)
        log_probs = torch.log(action_probs.gather(-1, torch.from_numpy(action).to(self.policy.device).unsqueeze(-1)))
        return action, log_probs

    def learn(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            G_t = 0
            pw = 0
            for r in rewards[t:]:
                G_t = G_t + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(G_t)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.policy.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, G_t in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob.squeeze() * G_t)

        self.policy.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()
        self.steps += 1