from tqdm import tqdm
import numpy as np
import torch
from utils import EpsRule

class Reinforce(object):

    def __init__(self, agent, env, dloader):
        super(Reinforce, self).__init__()
        self.agent = agent
        self.env = env
        self.dloader = dloader

    def test(self):
        self.agent.policy.eval()
        self.agent.reset_eps(0)
        with torch.no_grad():
            state = self.env.state
            while not self.env.done:
                action, log_prob = self.agent.get_action(state, by_max=True)
                state_, reward, keep_old_state = self.env.execute_action(action)
                # state = state_
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
            eps_rule = EpsRule(initial_eps=1, episode_shrinkage=1/(n_iterations/5), step_increase=0.1,
                               limiting_epsiode=n_iterations-10, change_after_n_episodes=5)
            self.agent.policy.train()

            print("----Fnished mem init----")
            for episode in tqdm(range(n_iterations)):
                log_probs = []
                rewards = []
                steps = 0
                state = self.env.state.copy()

                while not self.env.done:
                    self.agent.reset_eps(eps_rule.apply(episode, steps))
                    action, log_prob = self.agent.get_action(state)
                    state_, reward, _ = self.env.execute_action(action)
                    state = state_
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    steps += 1

                self.agent.learn(rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))

                raw, affinities, gt_affinities = next(iter(self.dloader))
                affinities = affinities.squeeze().detach().cpu().numpy()
                gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
                self.env.update_data(affinities, gt_affinities)

                scores.append(self.env.acc_reward)
                print("score: ", scores[-1], "; eps: ", self.agent.eps, "; steps: ", self.env.ttl_cnt)
                if showInterm:
                    self.env.show_current_soln()
                self.env.reset()
        return scores, steps, self.env.get_current_soln()