from agents.replayMemory import ReplayMemory


class QlAgent(object):

    def __init__(self, gamma=1, eps=0.9, eps_min=0.000001, replace_cnt=2, mem_size=10):
        super(QlAgent, self).__init__()
        self.gamma = gamma
        self.init_eps = eps
        self.eps = self.init_eps
        self.eps_min = eps_min

        self.mem = ReplayMemory(capacity=mem_size)
        self.learn_steps = 0
        self.steps = 0
        self.replace_cnt = replace_cnt

    def reset_eps(self, eps):
        self.eps = eps

    def safe_models(self, directory):
        return

    def load_model(self, directory):
        return

    def store_transit(self, state, action, reward, next_state, terminal):
        self.mem.push(state, action, reward, next_state, terminal)

    def get_action(self, state):
        return

    def learn(self, batch_size):
        pass