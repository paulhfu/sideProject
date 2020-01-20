from tqdm import tqdm
import numpy as np
from utils import EpsRule, ActionPathTreeNodes
import torch

class Qlearning(object):

    def __init__(self, agent, env, dloader=None, eps=0.9):
        super(Qlearning, self).__init__()
        self.agent = agent
        self.env = env
        self.eps = eps
        self.dloader = dloader

    def test(self, showInterm=True):
        # self.agent.q_eval.eval()
        self.agent.reset_eps(0)
        with torch.no_grad():
            state = self.env.state
            while not self.env.done:
                action = self.agent.get_action(state)
                state_, reward, keep_old_state = self.env.execute_action(action, learn=False)
                print("reward: ", reward)
                if showInterm:
                    self.env.show_current_soln()
                if keep_old_state:
                    self.env.state = state.copy()
                else:
                    state = state_
        return self.env.get_current_soln()

    def train_eps_greedy(self, n_iterations=150, batch_size=1, showInterm=True):
        with torch.set_grad_enabled(True):
            state = self.env.state
            eps_rule = EpsRule(initial_eps=1, episode_shrinkage=1 / (n_iterations / 5), step_increase=0.001,
                               limiting_epsiode=n_iterations - 10, change_after_n_episodes=5)
            self.agent.q_eval.train()
            for step in tqdm(range(self.agent.mem.capacity)):
                action = self.agent.get_action(state)
                state_, reward, keep_old_state = self.env.execute_action(action)
                self.agent.store_transit(state, action, reward, state_, int(self.env.done))
                if keep_old_state:
                    self.env.state = state.copy()
                else:
                    state = state_
                if self.env.done:
                    self.env.reset()
                    state = self.env.state

            print("----Fnished mem init----")
            eps_hist = []
            scores = []
            for episode in tqdm(range(n_iterations)):
                eps_hist.append(self.agent.eps)
                state = self.env.state.copy()
                steps = 0
                while not self.env.done:
                    self.agent.reset_eps(eps_rule.apply(episode, steps))
                    action = self.agent.get_action(state)
                    state_, reward, keep_old_state = self.env.execute_action(action)
                    self.agent.store_transit(state.copy(), action, reward, state_.copy(), int(self.env.done))
                    if keep_old_state:
                        self.env.state = state.copy()
                    else:
                        state = state_
                    # print(f'reward:{reward[1]}')
                    self.agent.learn(batch_size)
                    steps += 1
                if showInterm:
                    self.env.show_current_soln()
                if self.dloader is not None:
                    raw, affinities, gt_affinities = next(iter(self.dloader))
                    affinities = affinities.squeeze().detach().cpu().numpy()
                    gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
                    self.env.update_data(affinities, gt_affinities)
                scores.append(self.env.acc_reward)
                print("score: ", self.env.acc_reward, "; eps: ", self.agent.eps, "; steps: ", steps)
                self.env.reset()
        return scores, eps_hist, self.env.get_current_soln()

    def train_tree_search(self, n_iterations=150, batch_size=1, tree_node_weight=20, showInterm=True):
        with torch.set_grad_enabled(True):
            state = self.env.state
            ttl_steps = 0
            tree_nodes = ActionPathTreeNodes()
            self.agent.q_eval.train()
            for step in tqdm(range(self.agent.mem.capacity)):
                action = self.agent.get_action(state)
                state_, reward, keep_old_state = self.env.execute_action(action)
                self.agent.store_transit(state, action, reward, state_, int(self.env.done))
                if keep_old_state:
                    self.env.state = state.copy()
                else:
                    state = state_
                if self.env.done:
                    self.env.reset()
                    state = self.env.state

            print("----Fnished mem init----")
            eps_hist = []
            scores = []
            for episode in tqdm(range(n_iterations)):
                eps_hist.append(self.agent.eps)
                state = self.env.state.copy()
                steps = 0
                action_path = []
                while not self.env.done:
                    tree_nodes.push_path(action_path)
                    node_eps = max(1 - (tree_nodes.get_n_visits(action_path) / tree_node_weight), 0)
                    self.agent.reset_eps(node_eps)
                    action = self.agent.get_action(state)
                    state_, reward, keep_old_state = self.env.execute_action(action)
                    if node_eps == 0 and reward == self.env.penalty_reward:
                        tree_nodes.set_n_visits(action_path, 1)
                    else:
                        self.agent.store_transit(state.copy(), action, reward, state_.copy(), int(self.env.done))
                        self.agent.learn(batch_size)
                    action_path.append(action)
                    state = state_
                    steps += 1
                if showInterm:
                    self.env.show_current_soln()
                if self.dloader is not None:
                    raw, affinities, gt_affinities = next(iter(self.dloader))
                    affinities = affinities.squeeze().detach().cpu().numpy()
                    gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
                    self.env.update_data(affinities, gt_affinities)
                scores.append(self.env.acc_reward)
                print("score: ", self.env.acc_reward, "; eps: ", self.agent.eps, "; steps: ", steps)
                self.env.reset()
        return scores, eps_hist, self.env.get_current_soln()