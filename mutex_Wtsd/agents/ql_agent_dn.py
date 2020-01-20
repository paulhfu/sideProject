from models.ril_function_models import DNDQN, UnetFcnDQN, UnetDQN
from agents.qlagent import QlAgent
from agents.replayMemory import Transition
import numpy as np
import torch
import os


class QlAgentDN(QlAgent):
    """Agent for Q learning using DenseNet for Q-func"""

    def __init__(self, gamma, n_state_channels, n_actions, n_sel_mtxs, device,
                 eps=0.9, eps_min=0.000001, replace_cnt=2, mem_size=1,  writer=None):
        super(QlAgentDN, self).__init__(gamma=gamma, eps=eps, eps_min=eps_min, replace_cnt=replace_cnt, mem_size=mem_size)

        self.writer = writer

        self.q_eval = DNDQN(num_classes=n_actions+n_sel_mtxs, num_inchannels=n_state_channels, device=device)
        self.q_next = DNDQN(num_classes=n_actions+n_sel_mtxs, num_inchannels=n_state_channels, device=device)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = n_actions
        self.n_sel_mtxs = n_sel_mtxs
        self.n_state_channels = n_state_channels

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'dn_Q'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'dn_Q')), strict=True)
        self.q_next.load_state_dict(torch.load(os.path.join(directory, 'dn_Q')), strict=True)

    def get_action(self, state):
        if np.random.random() < (1-self.eps):
            self.q_eval.eval()
            q_vals = self.q_eval(torch.tensor(state, dtype=torch.float32).to(self.q_eval.device).unsqueeze(0)).squeeze()
            action = [torch.argmax(q_vals[:self.n_sel_mtxs], dim=0).item(),
                      torch.argmax(q_vals[self.n_sel_mtxs:], dim=0).item()]
        else:
            action = [np.random.randint(0, self.n_sel_mtxs),
                      np.random.randint(0, self.n_actions)]
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
            actions = torch.tensor(list(batch.action), dtype=torch.long).to(self.q_eval.device)
            actions[:, 1] += self.n_sel_mtxs
            rewards = torch.tensor(list(batch.reward), dtype=torch.float32).to(self.q_eval.device)

            predQ = self.q_eval(state).gather(1, actions)
            nextQ = self.q_next(next_state)
            nextQ = torch.cat([nextQ[:, :self.n_sel_mtxs].max(1)[0].detach().unsqueeze(1),
                               nextQ[:, self.n_sel_mtxs:].max(1)[0].detach().unsqueeze(1)], dim=1)

            tgtQ = (nextQ * self.gamma) + rewards.unsqueeze(1)

            loss = self.q_eval.loss(tgtQ, predQ)
            if self.writer is not None:
                self.writer.add_scalar("learn_step/loss", loss.item(), self.learn_steps)
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            for param in self.q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.q_eval.optimizer.step()
            self.learn_steps += 1
