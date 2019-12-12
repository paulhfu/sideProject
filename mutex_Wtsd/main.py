import os

from werkzeug.utils import environ_property

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from affogato.segmentation.mws import get_valid_edges
from affogato.segmentation import compute_mws_segmentation
from affogato.affinities import compute_affinities
from affogato.segmentation.mws import get_valid_edges
from q_learning import Agent, Mtx_wtsd_env
from simple_unet import UNet, smallUNet
from datasets import DiscDset, computeAffs, simpleSeg_4_4_Dset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import copy
import torch
import torch.nn as nn
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from matplotlib import cm

assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.FloatTensor)
from mutex_watershed import compute_mws_prim_segmentation

import h5py
import os
import numpy as np

# offsets = [[-1, 0], [0, -1], [-1, -1], [1, -1],
#            # direct 3d nhood for attractive edges
#            [-9, 0], [0, -9], [-9, -9], [9, -9],
#            # inplane diagonal dam edges
#            [-15, 0], [0, -15], [-15, -15], [15, -15]]
strides = []
separating_channel = 2
offsets = [[0, -1], [-1, 0],
           # direct 3d nhood for attractive edges
           [0, -1], [-1, 0]]
           # inplane diagonal dam edges
           # [-15, 0], [0, -15]]

def q_testing(affinities_predictor, dloader, rootPath):
    ril_lr = 0.001
    ril_disc = 1
    n_actions = 2
    raw, affinities = next(iter(dloader))
    affinities = affinities.squeeze().detach().cpu().numpy()
    affinities[separating_channel:] *= -1
    affinities[separating_channel:] += +1
    with torch.set_grad_enabled(False):
        affs = affinities_predictor(raw.to(device))
        affs = affs.squeeze().detach().cpu().numpy()
        affs[separating_channel:] *= -1
        affs[separating_channel:] += +1
        plt.imshow(raw.detach().cpu().squeeze().numpy());plt.show()

    agent = Agent(ril_lr, ril_disc, len(offsets)+separating_channel, 2, n_actions, 20, device, eps=0, eps_min=0)
    agent.load_model(rootPath)
    mtx_env = Mtx_wtsd_env(affs, separating_channel, 10, offsets, None,
                           affinities)
    print('---starting game---')
    state = mtx_env.state
    last_rew = -100
    while not mtx_env.done:
        action = agent.get_action(state)
        state_, reward = mtx_env.execute_action(action)
        # state = state_
        if reward >= last_rew:
            state = state_
        else:
            mtx_env.state = state.copy()
        last_rew = reward
        print(reward)
        print(action)
    mtx_env.show_current_soln()

def q_learning(affinities_predictor, dloader, rootPath):
    ril_lr = 0.001
    ril_disc = 1
    n_actions = 2
    eps=0.9
    n_mtx_offs = len(offsets)
    agent = Agent(ril_lr, ril_disc, len(offsets)+separating_channel, 2, n_actions, 20, device, eps=eps)
    agent.load_model(rootPath)
    # agent.load_model(rootPath)
    raw, affinities = next(iter(dloader))
    # affinities, _ = compute_affinities(raw.detach().cpu().squeeze().numpy(), offsets)
    affinities = affinities.squeeze().detach().cpu().numpy()
    affinities[separating_channel:] *= -1
    affinities[separating_channel:] += +1
    valid_edges = get_valid_edges(affinities.shape, offsets, separating_channel, strides=None, randomize_strides=False)
    # node_labeling = compute_mws_prim_segmentation(affinities.ravel(),
    #                                               valid_edges.ravel(),
    #                                               offsets,
    #                                               separating_channel,
    #                                               [4, 4])
    # plt.imshow(node_labeling.reshape(affinities.shape[1:]));plt.show()
    # affinities_predictor.eval()
    # affinities_predictor.cuda()
    with torch.set_grad_enabled(False):
        affs = affinities_predictor(raw.to(device))
        affs = affs.squeeze().detach().cpu().numpy()
        affs[separating_channel:] *= -1
        affs[separating_channel:] += +1
        node_labeling = compute_mws_prim_segmentation(affs.ravel(),
                                                      valid_edges.ravel(),
                                                      offsets,
                                                      separating_channel,
                                                      [4,4])
        labels = node_labeling.reshape([4,4])
        # show_seg = cm.prism(labels / labels.max())
        # plt.imshow(show_seg);plt.show()
        plt.imshow(raw.detach().cpu().squeeze().numpy());plt.show()
    # (self, edge_predictor, raw_image, separating_channel, offsets, strides, gt_affinities, use_bbox=False)
    mtx_env = Mtx_wtsd_env(affs, separating_channel, 10, offsets, None,
                           affinities.astype(np.float))
    state = mtx_env.state
    for i in tqdm(range(agent.mem.capacity)):
        action = agent.get_action(state)
        state_, reward = mtx_env.execute_action(action)
        agent.store_transit(state, action, reward, state_)
        state = state_
        if mtx_env.done:
            mtx_env.reset()
            state = mtx_env.state
        if mtx_env.done:
            mtx_env.reset()
    print("----Fnished mem init----")
    n_iterations = 100
    batch_size = 8
    eps_hist = []
    scores = []
    for i in tqdm(range(n_iterations)):
        print('---starting game---')
        last_rew = -100
        eps_hist.append(agent.eps)
        state = mtx_env.state
        while not mtx_env.done:
            action = agent.get_action(state)
            state_, reward = mtx_env.execute_action(action)
            agent.store_transit(state, action, reward, state_)
            # state = state_
            if reward >= last_rew:
                state = state_
            else:
                mtx_env.state = state.copy()
            last_rew = reward
            print(f'reward:{reward}')
            # print(action)
            agent.learn(batch_size)
        scores.append(mtx_env.accumulated_reward)
        mtx_env.reset()
        if i % 5 == 0:
            agent.reset_eps(eps)
            eps -= 0.1
            if i > 60:
                eps = 0
    mtx_env.show_current_soln()
    agent.safe_model(rootPath)


def trainAffPredSimpleImg(saveToFile, numEpochs=2):
    dloader = DataLoader(simpleSeg_4_4_Dset(), batch_size=1, shuffle=True, pin_memory=True)

    print('----START TRAINING----' * 4)

    model = smallUNet(n_channels=1, n_classes=len(offsets), bilinear=True)
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()

    optim = torch.optim.SGD(model.parameters(), lr=0.18)

    model.cuda()
    since = time.time()

    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Iterate over data.
        for step, (inputs, affinities) in enumerate(dloader):
            inputs = inputs.to(device)
            affinities = affinities.to(device)
            # zero the parameter gradients
            optim.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, affinities)
                loss.backward()
                optim.step()

        weights = outputs.squeeze().detach().cpu().numpy()
        affs = affinities.squeeze().detach().cpu().numpy()
        weights[separating_channel:] *= -1
        weights[separating_channel:] += +1
        affs[separating_channel:] *= -1
        affs[separating_channel:] += +1
        outputs = model(inputs)

        ndim = len(offsets[0])
        # assert all(len(off) == ndim for off in offsets)
        # image_shape = weights.shape[1:]
        # valid_edges = get_valid_edges(weights.shape, offsets, separating_channel, strides=None, randomize_strides=False)
        # node_labeling0 = compute_mws_prim_segmentation(weights.ravel(),
        #                                               valid_edges.ravel(),
        #                                               offsets,
        #                                               separating_channel,
        #                                               image_shape)
        # node_labeling1 = compute_mws_prim_segmentation(affs.ravel(),
        #                                               valid_edges.ravel(),
        #                                               offsets,
        #                                               separating_channel,
        #                                               image_shape)
        # node_labeling = compute_mws_segmentation(affs, offsets, separating_channel, algorithm='prim')
        # labels = node_labeling.reshape(image_shape)
        # labels1 = node_labeling1.reshape(image_shape)
        # labels0 = node_labeling0.reshape(image_shape)
        # show_seg1 = cm.prism(labels1 / labels1.max())
        # show_seg = cm.prism(labels / labels.max())
        # show_seg0 = cm.prism(labels0 / labels0.max())
        # show_raw = cm.gray(inputs.squeeze().detach().cpu().numpy())
        # plt.imshow(show_seg0);
        # plt.show()
        # for i in range(1):
        #     img = np.concatenate([np.concatenate([show_seg, show_seg1], axis=1),
        #                           np.concatenate([show_raw, show_seg0], axis=1)], axis=0)
        #     plt.imshow(img);plt.show()
        #
        # torch.save(model.state_dict(), saveToFile)

    return model


def trainAffPredCircles(saveToFile, numEpochs=3):
    file = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Discs'

    dloader = DataLoader(DiscDset(rootPath, file, mode='train'), batch_size=1, shuffle=True, pin_memory=True)

    print('----START TRAINING----' * 4)

    model = UNet(n_channels=1, n_classes=len(offsets), bilinear=True)
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()

    optim = torch.optim.SGD(model.parameters(),  lr=0.01)

    model.cuda()
    since = time.time()

    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Iterate over data.
        for step, (inputs, affinities) in enumerate(dloader):
            inputs = inputs.to(device)
            affinities = affinities.to(device)
            # zero the parameter gradients
            optim.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, affinities)
                loss.backward()
                optim.step()

        weights = outputs.squeeze().detach().cpu().numpy()
        affs = affinities.squeeze().detach().cpu().numpy()
        weights[separating_channel:] *= -1
        weights[separating_channel:] += +1
        affs[separating_channel:] *= -1
        affs[separating_channel:] += +1

        ndim = len(offsets[0])
        assert all(len(off) == ndim for off in offsets)
        image_shape = weights.shape[1:]
        valid_edges = get_valid_edges(weights.shape, offsets, separating_channel, strides, False)
        node_labeling1 = compute_mws_prim_segmentation(weights.ravel(),
                                                      valid_edges.ravel(),
                                                      offsets,
                                                      separating_channel,
                                                      image_shape)
        node_labeling = compute_mws_segmentation(affs, offsets, separating_channel, strides=strides, randomize_strides=True, algorithm='prim')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        labels = node_labeling.reshape(image_shape)
        labels1 = node_labeling1.reshape(image_shape)
        show_seg1 = cm.prism(labels1 / labels1.max())
        show_seg = cm.prism(labels / labels.max())
        show_raw = cm.gray(inputs.squeeze().detach().cpu().numpy())
        for i in range(1):
            img = np.concatenate([np.concatenate([show_seg, show_seg1], axis=1),
                                  np.concatenate([show_raw, cm.gray(weights[i])], axis=1)], axis=0)
            plt.imshow(img);plt.show()

    torch.save(model.state_dict(), saveToFile)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return


if __name__ == '__main__':
    files = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Discs'
    modelFile = os.path.join(rootPath, 'UnetEdgePredsSimple.pth')
    dloader = DataLoader(simpleSeg_4_4_Dset(), batch_size=1, shuffle=True, pin_memory=True)
    affinities_predictor = smallUNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    affinities_predictor.load_state_dict(torch.load(modelFile), strict=True)
    affinities_predictor.cuda()
    # computeAffs(os.path.join(rootPath, files), offsets)
    affinities_predictor = trainAffPredSimpleImg(modelFile)
    # q_learning(affinities_predictor, dloader, rootPath)
    q_testing(affinities_predictor, dloader, rootPath)
    # trainAffPred(modelFile)
    # trainAffPredSimpleImg(modelFile)