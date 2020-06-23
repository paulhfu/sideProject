import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, '/g/kreshuk/hilt/projects/fewShotLearning/mu-net')
import torch
import hydra
import dmc2gym



if __name__ == '__main__':
    env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)

    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)