from tqdm import tqdm
import numpy as np
import torch
from agents.exploitation_functions import NaiveDecay, ActionPathTreeNodes, ExpSawtoothEpsDecay

class A2c(object):

    def __init__(self, agent, env, dloader):
        super(A2c, self).__init__()
        self.agent = agent
        self.env = env
        self.dloader = dloader

    def test(self):
        self.agent.policy.eval()
        self.agent.reset_eps(0)
        with torch.no_grad():
            state = self.env.state
            while not self.env.done:
                action, _, _, _ = self.agent.get_action(state, by_max=True)
                state_, reward, keep_old_state = self.env.execute_action(action)
                # state = state_
                print("reward: ", reward)
                self.env.show_current_soln()
                if keep_old_state:
                    self.env.state = state.copy()
                else:
                    state = state_
        return self.env.get_current_soln()

    def train(self, n_iterations=150, showInterm=False):
        with torch.set_grad_enabled(True):
            numsteps = []
            avg_numsteps = []
            scores = []
            self.agent.policy.train()
            eps_rule = NaiveDecay(initial_eps=1, episode_shrinkage=1 / (n_iterations / 5),
                               limiting_epsiode=n_iterations - 10, change_after_n_episodes=5)

            print("----Fnished mem init----")
            for episode in tqdm(range(n_iterations)):
                log_probs = []
                rewards = []
                values = []
                policy_entropy = 0
                steps = 0
                state = self.env.state

                while not self.env.done:
                    self.agent.reset_eps(eps_rule.apply(episode, steps))
                    action, log_prob, value, entropy = self.agent.get_action(state)
                    state_, reward, _ = self.env.execute_action(action)
                    state = state_
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    values.append(value)
                    policy_entropy += entropy
                    steps += 1

                _, _, q_val, _ = self.agent.get_action(state)
                self.agent.learn(rewards, log_probs, values, policy_entropy, q_val)

                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))

                raw, affinities, gt_affinities = next(iter(self.dloader))
                affinities = affinities.squeeze().detach().cpu().numpy()
                gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
                scores.append(self.env.acc_reward)
                print("score: ", scores[-1], "; eps: ", self.agent.eps, "; steps: ", self.env.ttl_cnt)
                self.env.update_data(affinities, gt_affinities)
                self.env.reset()

                if showInterm:
                    self.env.show_current_soln()

        return scores, steps, self.env.get_current_soln()
