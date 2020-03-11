from tqdm import tqdm
import numpy as np
from agents.exploitation_functions import NaiveEpsDecay, ActionPathTreeNodes, ExpSawtoothEpsDecay
import torch

class Qlearning(object):

    def __init__(self, agent, env, dloader, eps=0.9):
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
            eps_rule = NaiveEpsDecay(initial_eps=1, episode_shrinkage=1 / (n_iterations / 5),
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
            # fill memory
            for step in tqdm(range(self.agent.mem.capacity)):
                action = self.agent.get_action(state)
                state_, reward, keep_old_state = self.env.execute_action(action)
                self.agent.store_transit(state, action, reward, state_, int(self.env.done))
                if keep_old_state:
                    self.env.state = state.copy()
                else:
                    state = state_
                if self.env.donenp.exp(-(0.0003 * episode) ** 2) * 20:
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
                    tree_nodes.push_path("".join(map(str, action_path)))
                    node_eps = max(1 - (tree_nodes.get_n_visits("".join(map(str, action_path))) / tree_node_weight), 0)
                    self.agent.reset_eps(node_eps)
                    action = self.agent.get_action(state)
                    state_, reward, keep_old_state = self.env.execute_action(action)
                    if node_eps == 0 and reward == self.env.penalty_reward:
                        tree_nodes.set_n_visits("".join(map(str, action_path)), 1)
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

    def train_retrace_gcn(self, n_iterations=150, limiting_behav_iter=130):
        with torch.set_grad_enabled(True):
            self.agent.q_eval.train()
            eps_rule = ExpSawtoothEpsDecay(initial_eps=1, episode_shrinkage=1 / (n_iterations / 5), step_increase=n_iterations * 0.001,
                               limiting_epsiode=limiting_behav_iter, change_after_n_episodes=5)

            print("----Fnished mem init----")
            eps_hist = []
            scores = []
            for episode in tqdm(range(n_iterations)):

                edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles = next(iter(self.dloader))
                edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles = \
                    edges.squeeze(), edge_feat.squeeze(), diff_to_gt.squeeze(), gt_edge_weights.squeeze(), \
                    node_labeling.squeeze(), raw.squeeze(), nodes.squeeze(), angles.squeeze()
                self.env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles)

                eps_hist.append(self.agent.eps)
                state = self.env.state.clone()
                steps = 0
                eps_steps = 0
                while not self.env.done:
                    self.agent.reset_eps(eps_rule.apply(episode, eps_steps))
                    self.env.stop_quality = np.exp(-(0.0015 * episode) ** 2) * 5
                    # self.agent.reset_eps(0)
                    self.env.stop_quality = 3
                    if self.agent.eps == 0:
                        eps_steps = 0
                    action, behav_probs = self.agent.get_action(state, self.env.counter)
                    state_, reward = self.env.execute_action(action, episode)
                    self.agent.store_transit(state.clone(), action, reward, state_.clone(), steps, behav_probs, int(self.env.done))
                    if not len(self.agent.mem) < self.agent.mem.cap:
                        self.agent.learn()
                    state = state_
                    steps += 1
                    eps_steps += 1
                while len(self.agent.mem) > 0:
                    self.agent.learn()
                    self.agent.mem.pop(0)
                self.agent.mem.clear()
                scores.append(self.env.acc_reward)
                print("score: ", self.env.acc_reward, "; eps: ", self.agent.eps, "; steps: ", steps)
                self.env.reset()
        return scores, eps_hist, None
