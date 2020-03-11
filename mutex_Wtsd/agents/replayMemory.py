from collections import namedtuple
import random
import numpy as np

Transition_t = namedtuple('Transition', ('state', 'actions', 'reward', 'state_', 'time', 'behav_probs', 'terminal'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TransitionData(object):

    def __init__(self, capacity=20):
        self.memory = []
        self.position = 0
        self.cap = capacity

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if self.position >= self.cap:
            self.pop(0)
        self.memory.append(None)
        self.memory[self.position] = Transition_t(*args)
        self.position += 1

    def pop(self, position):
        self.position -= 1
        return self.memory.pop(position)

    def sample(self, batch_size, distribution=None):
        return np.random.choice(self.memory, size=batch_size, p=distribution)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)
