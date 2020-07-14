from collections import namedtuple
import random
import numpy as np
import torch

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


class TransitionData_ts(object):

    def __init__(self, capacity, storage_object):
        self.memory = []
        self.position = 0
        self.cap = capacity
        self.storage_object = storage_object
        self.sampled_n_times = []

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if self.position >= self.cap:
            drop_out = self.sampled_n_times.index(max(self.sampled_n_times))
            self.pop(drop_out)
        self.memory.append(None)
        self.sampled_n_times.append(1)
        self.memory[self.position] = self.storage_object(*args)
        self.position += 1

    def is_full(self):
        if self.position >= self.cap:
            return True
        return False

    def pop(self, position):
        self.position -= 1
        self.sampled_n_times.pop(position)
        return self.memory.pop(position)

    def sample(self):
        distribution = torch.softmax(1/(torch.tensor(self.sampled_n_times, dtype=torch.float)), 0)
        sample_idx = torch.multinomial(distribution, 1).item()
        self.sampled_n_times[sample_idx] += 1
        return self.memory[sample_idx]

    def clear(self):
        self.memory = []
        self.sampled_n_times = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class TransitionData(object):

    def __init__(self, capacity, storage_object):
        self.memory = []
        self.position = 0
        self.cap = capacity
        self.storage_object = storage_object

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if self.position >= self.cap:
            self.pop(0)
        self.memory.append(None)
        self.memory[self.position] = self.storage_object(*args)
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


# -*- coding: utf-8 -*-
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))


class EpisodeTrajectoriesMem():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity
    self.memory = deque(maxlen=self.num_episodes)
    self.trajectory = []

  def append(self, state, action, reward, policy):
    self.trajectory.append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if action is None:
      self.memory.append(self.trajectory)
      self.trajectory = []
  # Samples random trajectory
  def sample(self, maxlen=0):
    mem = self.memory[random.randrange(len(self.memory))]
    T = len(mem)
    # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
    if maxlen > 0 and T > maxlen + 1:
      t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
      return mem[t:t + maxlen + 1]
    else:
      return mem

  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def length(self):
    # Return number of epsiodes saved in memory
    return len(self.memory)

  def __len__(self):
    return sum(len(episode) for episode in self.memory)