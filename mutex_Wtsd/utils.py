import numpy as np
import torch

def ind_flat_2_spat(flat_indices, shape):
    spat_indices = np.zeros([len(flat_indices)] + [len(shape)], dtype=np.integer)
    for flat_ind, spat_ind in zip(flat_indices, spat_indices):
        rm = flat_ind
        for dim in range(1, len(shape)):
            sz = np.prod(shape[dim:])
            spat_ind[dim - 1] = rm // sz
            rm -= spat_ind[dim - 1] * sz
        spat_ind[-1] = rm
    return spat_indices

def ind_spat_2_flat(spat_indices, shape):
    flat_indices = np.zeros(len(spat_indices), dtype=np.integer)
    for i, spat_ind in enumerate(spat_indices):
        for dim in range(len(shape)):
            flat_indices[i] += max(1, np.prod(shape[dim + 1:])) * spat_ind[dim]
    return flat_indices

def add_rndness_in_dis(dis, factor):
    assert isinstance(dis, np.ndarray)
    assert len(dis.shape) == 2
    ret_dis = dis - ((dis - np.transpose([np.mean(dis, axis=-1)])) * factor)
    return dis


class EpsRule(object):

    def __init__(self, initial_eps, episode_shrinkage, step_increase, limiting_epsiode, change_after_n_episodes):
        self.initial_eps = initial_eps
        self.episode_shrinkage = episode_shrinkage
        self.step_increase = step_increase
        self.limiting_epsiode = limiting_epsiode
        self.change_after_n_episodes = change_after_n_episodes

    def apply(self, episode, step):
        if episode >= self.limiting_epsiode:
            return 0
        eps = self.initial_eps-(self.episode_shrinkage * (episode//self.change_after_n_episodes))
        eps = eps + (self.step_increase * step)
        eps = min(max(eps, 0), 1)
        return eps

class ActionPathTreeNodes(object):

    def __init__(self):
        self.memory = {}

    def push_path(self, path):
        if path:
            key = int("".join(map(str, path)))
        else:
            key = "first"
        if key in self.memory:
            self.memory[key] += 1
        else:
            self.memory[key] = 1

    def get_n_visits(self, path):
        if path:
            key = int("".join(map(str, path)))
        else:
            key = "first"
        if key in self.memory:
            return self.memory[key]
        else:
            return 0

    def set_n_visits(self, path, visits):
        if path:
            key = int("".join(map(str, path)))
        else:
            key = "first"
        self.memory[key] = visits

    def clear_memory(self):
        self.memory = {}
