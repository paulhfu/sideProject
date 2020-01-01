from tqdm import tqdm

def eps_rule1(itr, eps):
    if itr % 5 == 0:
        eps -= 0.01
    if itr > 400:
        eps = 0
    return eps

def eps_rule2(itr, eps):
    """used this for very small images"""
    if itr % 50 == 0:
        eps = max(eps-0.01, 0)
    if itr > 3500:
        eps = 0
    return eps

class Qlearning(object):

    def __init__(self, agent, env, eps=0.9):
        super(Qlearning, self).__init__()
        self.agent = agent
        self.env = env
        self.eps = eps

    def test(self):
        state = self.env.state
        while not self.env.done:
            action = self.agent.get_action(state)
            state_, reward, keep_old_state = self.env.execute_action(action)
            # state = state_
            if keep_old_state:
                self.env.state = state.copy()
            else:
                state = state_
        return self.env.get_current_soln()

    def train(self, n_iterations=150, batch_size=1, showInterm=False):
        state = self.env.state
        for i in tqdm(range(self.agent.mem.capacity)):
            action = self.agent.get_action(state)
            state_, reward, keep_old_state = self.env.execute_action(action)
            self.agent.store_transit(state, action, reward, state_)
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
        for i in tqdm(range(n_iterations)):
            print('---starting game---')
            last_rew = -100
            eps_hist.append(self.agent.eps)
            state = self.env.state.copy()
            while not self.env.done:
                action = self.agent.get_action(state)
                state_, reward, keep_old_state = self.env.execute_action(action)
                self.agent.store_transit(state.copy(), action.copy(), reward.copy(), state_.copy())
                if keep_old_state:
                    self.env.state = state.copy()
                else:
                    state = state_
                # print(f'reward:{reward[1]}')
                self.agent.learn(batch_size)
            scores.append(self.env.acc_reward)
            if showInterm:
                self.env.show_current_soln()
            self.env.reset()
            self.eps = eps_rule2(i, self.eps)
            self.agent.reset_eps(self.eps)
        return scores, eps_hist, self.env.get_current_soln()
