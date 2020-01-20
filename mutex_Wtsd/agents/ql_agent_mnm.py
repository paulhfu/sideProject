from models.ril_function_models import DNDQN, UnetFcnDQN, UnetDQN
from agents.qlagent import QlAgent
from matplotlib import cm
import matplotlib.pyplot as plt
from agents.replayMemory import Transition
import numpy as np
import torch
import os


class QlAgentMNM(QlAgent):
    """Agent for Q learning using DenseNet for Q-func"""

    def __init__(self, gamma, n_state_channels, n_actions, device,
                 eps=0.9, eps_min=0.000001, replace_cnt=5, mem_size=10,  writer=None):
        super(QlAgentMNM, self).__init__(gamma=gamma, eps=eps, eps_min=eps_min, replace_cnt=replace_cnt, mem_size=mem_size)

        self.writer = writer

        self.q_eval = DNDQN(num_classes=n_actions, num_inchannels=n_state_channels, device=device, block_config=(6,))
        self.q_next = DNDQN(num_classes=n_actions, num_inchannels=n_state_channels, device=device, block_config=(6,))
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = n_actions
        self.n_state_channels = n_state_channels

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'mnm_simple_Q'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'mnm_simple_Q')), strict=True)
        self.q_next.load_state_dict(torch.load(os.path.join(directory, 'mnm_simple_Q')), strict=True)

    def get_action(self, state):
        # img1 = np.concatenate([cm.prism(state[0] / state[0].max()), cm.prism(state[1] / state[1].max())], axis=0);plt.imshow(img1);plt.show();
        if np.random.random() < self.eps:
            action = np.random.randint(0, self.n_actions)
        else:
            self.q_eval.eval()
            q_vals = self.q_eval(torch.tensor(state, dtype=torch.float32).to(self.q_eval.device).unsqueeze(0)).squeeze()
            action = torch.argmax(q_vals, dim=0).item()
        self.steps += 1
        return action

    def learn(self, batch_size):
        with torch.set_grad_enabled(True):
            loss = 0
            self.q_eval.train()
            batch = self.mem.sample(batch_size)
            batch = Transition(*zip(*batch))
            state = torch.tensor(list(batch.state), dtype=torch.float32, device=self.q_eval.device)
            next_state = torch.tensor(list(batch.next_state), dtype=torch.float32, device=self.q_next.device)
            actions = torch.tensor(list(batch.action), dtype=torch.long, device=self.q_eval.device)
            rewards = torch.tensor(list(batch.reward), dtype=torch.float32, device=self.q_eval.device)
            terminal_state_mask = torch.tensor(list(batch.terminal), dtype=torch.float32, device=self.q_eval.device)

            if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
                self.q_next.load_state_dict(self.q_eval.state_dict())
            # tgt_finQ = torch.tensor([0, 0], dtype=torch.float32, device=self.q_eval.device)
            # finQ = self.q_eval(torch.zeros_like(state[0]).unsqueeze(0))
            # loss += self.q_eval.loss(tgt_finQ, finQ)
            # print("loss: ", loss.item())

            predQ = self.q_eval(state).gather(-1, actions.unsqueeze(-1))

            nextQ = self.q_next(next_state)
            nextQ = nextQ.max(-1)[0].detach().unsqueeze(1)


            tgtQ = (nextQ * self.gamma * terminal_state_mask.unsqueeze(1)) + rewards.unsqueeze(1)
            loss += self.q_eval.loss(tgtQ, predQ)
            if self.writer is not None:
                self.writer.add_scalar("learn_step/loss", loss.item(), self.learn_steps)
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            for param in self.q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.q_eval.optimizer.step()
            self.learn_steps += 1
