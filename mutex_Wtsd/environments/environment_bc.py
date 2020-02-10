class Environment(object):
    def __init__(self, stop_cnt=None):
        super(Environment, self).__init__()
        self.done = False
        self.counter = 0
        self.acc_reward = 0
        self.stop_cnt = stop_cnt

    def _update_state(self):
        return

    def execute_action(self, action):
        return

    def reset(self):
        return
