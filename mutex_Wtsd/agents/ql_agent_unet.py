from models.ril_function_models import DNDQN, UnetFcnDQN, UnetDQN
from agents.agent import Agent
from agents.replayMemory import Transition
import numpy as np
import torch
import os


class QlAgentUnet(Agent):
    """Agent for Q learning using Unet+FCN for Q-func"""

    def __init__(self, gamma, n_state_channels, n_edges, action_shape, device,
                 eps=0.9, eps_min=0.000001, replace_cnt=5, mem_size=40,  writer=None):
        super(QlAgentUnet, self).__init__(gamma=gamma, eps=eps, eps_min=eps_min, replace_cnt=replace_cnt, mem_size=mem_size)

        self.writer = writer
        self.action_shape = action_shape
        self.q_eval = UnetDQN(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=len(action_shape), device=device)
        self.q_next = UnetDQN(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=len(action_shape), device=device)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = len(action_shape)
        self.n_state_channels = n_state_channels

    def reset_eps(self, eps):
        self.eps = eps

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'unet_Q'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'unet_Q')), strict=True)
        self.q_next.load_state_dict(torch.load(os.path.join(directory, 'unet_Q')), strict=True)

    def get_action(self, state):
        self.q_eval.eval()
        q_vals_action = self.q_eval(torch.tensor(state, dtype=torch.float32).to(self.q_eval.device).unsqueeze(0))
        action = torch.argmax(q_vals_action.squeeze(), dim=1).cpu().numpy()
        sz = np.prod(self.action_shape)
        n_rnd_insertions = int(self.eps * sz)
        print("insertions: " + str(n_rnd_insertions))
        if n_rnd_insertions >= 1:
            indices = np.random.choice(np.arange(sz), n_rnd_insertions)
            action = action.ravel()
            action[indices] = np.random.randint(0, 2, len(indices))
            action = action.reshape(self.action_shape)
        self.steps += 1
        return action

    def learn(self, batch_size):
        with torch.set_grad_enabled(True):
            if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
                self.q_next.load_state_dict(self.q_eval.state_dict())

            batch = self.mem.sample(batch_size)
            batch = Transition(*zip(*batch))
            state = torch.tensor(list(batch.state), dtype=torch.float32).to(self.q_eval.device)
            next_state = torch.tensor(list(batch.next_state), dtype=torch.float32).to(self.q_next.device)
            action = torch.tensor(list(batch.action), dtype=torch.long).to(self.q_eval.device)
            reward = torch.tensor(list(batch.reward), dtype=torch.float32).to(self.q_eval.device)

            pred = self.q_eval(state)
            used_pred = pred.gather(2, action.unsqueeze(2)).squeeze()
            next_action = self.q_next(next_state).max(dim=2)[0]
            tgt = (next_action * self.gamma) + reward

            loss = self.q_eval.loss(tgt.squeeze(), used_pred)
            if self.writer is not None:
                self.writer.add_scalar("learn_step/loss", loss.item(), self.learn_steps)
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            for param in self.q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.q_eval.optimizer.step()
            self.learn_steps += 1


