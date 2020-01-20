from models.ril_function_models import DNDQN, UnetFcnDQN, UnetDQN
from agents.qlagent import QlAgent
from agents.replayMemory import Transition
import numpy as np
import torch
import os


class QlAgentUnetFcn(QlAgent):
    """Agent for Q learning using Unet+FCN for Q-func"""

    def __init__(self, gamma, n_state_channels, n_edges, n_actions, action_shape, device,
                 eps=0.9, eps_min=0.000001, replace_cnt=2, mem_size=1,  writer=None):
        super(QlAgentUnetFcn, self).__init__(gamma=gamma, eps=eps, eps_min=eps_min, replace_cnt=replace_cnt, mem_size=mem_size)

        self.writer = writer
        self.action_shape = action_shape
        self.q_eval = UnetFcnDQN(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=n_actions, device=device)
        self.q_next = UnetFcnDQN(n_inChannels=n_state_channels, n_edges=n_edges, n_actions=n_actions, device=device)
        self.q_eval.cuda(device=self.q_eval.device)
        self.q_next.cuda(device=self.q_eval.device)
        self.n_actions = n_actions
        self.n_state_channels = n_state_channels

    def reset_eps(self, eps):
        self.eps = eps

    def safe_model(self, directory):
        torch.save(self.q_eval.state_dict(), os.path.join(directory, 'unet_fcn_Q'))

    def load_model(self, directory):
        self.q_eval.load_state_dict(torch.load(os.path.join(directory, 'unet_fcn_Q')), strict=True)
        self.q_next.load_state_dict(torch.load(os.path.join(directory, 'unet_fcn_Q')), strict=True)

    def get_action(self, state):
        if np.random.random() < (1-self.eps):
            self.q_eval.eval()
            q_vals_selection, q_vals_action = self.q_eval(torch.tensor(state, dtype=torch.float32).to(self.q_eval.device).unsqueeze(0))
            action = (torch.argmax(q_vals_selection.squeeze(), dim=1).cpu().numpy(),
                      torch.argmax(q_vals_action.squeeze(), dim=0).item())
        else:
            action = (np.random.randint(0, 2, self.action_shape),
                      np.random.randint(0, self.n_actions))
        self.steps += 1
        return action

    def learn(self, batch_size):
        with torch.set_grad_enabled(True):
            self.q_eval.train()
            if self.replace_cnt is not None and self.learn_steps % self.replace_cnt == 0:
                self.q_next.load_state_dict(self.q_eval.state_dict())

            batch = self.mem.sample(batch_size)
            batch = Transition(*zip(*batch))
            state = torch.tensor(list(batch.state), dtype=torch.float32).to(self.q_eval.device)
            next_state = torch.tensor(list(batch.next_state), dtype=torch.float32).to(self.q_next.device)
            edge_selections = torch.tensor(list(np.array(batch.action)[:, 0]), dtype=torch.long).to(self.q_eval.device)
            action = torch.tensor(list(np.array(batch.action)[:, 1]), dtype=torch.long).to(self.q_eval.device)
            rewards_sel = torch.tensor(list(np.array(batch.reward)[:, 0]), dtype=torch.float32).to(self.q_eval.device)
            rewards_act = torch.tensor(list(np.array(batch.reward)[:, 1]), dtype=torch.float32).to(self.q_eval.device)

            pred_selection, pred_action = self.q_eval(state)
            pred_action_used = pred_action[torch.arange(pred_action.shape[0]), action]
            pred_selection_used = pred_selection.gather(2, edge_selections.unsqueeze(2)).squeeze()
            next_selection, next_action = self.q_next(next_state)
            next_action_used = next_action.max(dim=1)[0]
            next_selection_used = next_selection.max(dim=2)[0]

            tgt_selection = (next_selection_used * self.gamma) + rewards_sel
            tgt_action = (next_action_used * self.gamma) + rewards_act

            loss = self.q_eval.loss(tgt_selection, pred_selection_used) + self.q_eval.loss(tgt_action, pred_action_used)
            if self.writer is not None:
                self.writer.add_scalar("learn_step/loss", loss.item(), self.learn_steps)
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            for param in self.q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.q_eval.optimizer.step()
            self.learn_steps += 1

