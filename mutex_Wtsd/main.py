import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from train_affinities import trainAffPredCircles, trainAffPredSimpleImg
from mutex_watershed import compute_mws_segmentation_cstm
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation.mws import get_valid_edges
from q_learning import Qlearning
from agents.ql_agent_unet_fcn import QlAgentUnetFcn
from agents.ql_agent_dn import QlAgentDN
from agents.ql_agent_unet import QlAgentUnet
from environments.mtxwtsd_unet_fcn import MtxWtsdEnvUnetFcn
from environments.mtxwtsd_unet import MtxWtsdEnvUnet
from models.simple_unet import UNet, smallUNet
from data.datasets import simpleSeg_4_4_Dset, CustomDiscDset
from torch.utils.data import DataLoader
import time
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import cm

assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.FloatTensor)
from mutex_watershed import compute_partial_mws_prim_segmentation

import os
import numpy as np

# offsets = [[-1, 0], [0, -1], [-1, -1], [1, -1],
#            # direct 3d nhood for attractive edges
#            [-9, 0], [0, -9], [-9, -9], [9, -9],
#            # inplane diagonal dam edges
#            [-15, 0], [0, -15], [-15, -15], [15, -15]]
strides = None
separating_channel = 2
offsets = [[0, -1], [-1, 0],
           # direct 3d nhood for attractive edges
           # [0, -1], [-1, 0]] # this for simpleImg
           # inplane diagonal dam edges
           [-3, 0], [0, -3]]

def q_testing(affinities_predictor, dloader, rootPath):
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
        plt.imshow(raw.detach().cpu().squeeze().numpy());
        plt.show()

    agent = QlAgentUnet(gamma=1, n_state_channels=len(offsets)+separating_channel, n_edge_offs=len(offsets),
                           n_actions=2, action_shape=affs.shape, device=device)
    agent.load_model(rootPath)
    env = MtxWtsdEnvUnetFcn(affs=affs, separating_channel=separating_channel, offsets=offsets, strides=strides,
                            gt_affinities=affinities, device=device)
    # agent = QlAgentDN(gamma=ril_disc, n_state_channels=len(offsets)+separating_channel, n_mtxs=2, n_actions=n_actions, n_sel_mtxs=20, device=device, eps=0, eps_min=0)
    # agent.load_model(rootPath)
    # env = MtxWtsdEnvDN(affs=affs, separating_channel=separating_channel, strides=strides, separating_action=10, offsets=offsets, gt_affinities=affinities)
    ql = Qlearning(agent, env)
    final_labeling = ql.test()
    show_seg = cm.prism(final_labeling / final_labeling.max())
    plt.imshow(show_seg);plt.show()



def q_learning(affinities_predictor, dloader, rootPath):
    raw, affinities = next(iter(dloader))
    affinities = affinities.squeeze().detach().cpu().numpy()
    affinities[separating_channel:] *= -1
    affinities[separating_channel:] += +1
    with torch.set_grad_enabled(False):
        affs = affinities_predictor(raw.to(device))
        affs = affs.squeeze().detach().cpu().numpy()
        affs[separating_channel:] *= -1
        affs[separating_channel:] += +1
        node_labeling = compute_mws_segmentation(affs, offsets, separating_channel, algorithm='prim')
        # plt.imshow(cm.prism(node_labeling / node_labeling.max()))
        # plt.show()

    # agent = QlAgentDN(gamma=ril_disc, n_state_channels=len(offsets)+separating_channel, n_mtxs=2, n_actions=n_actions, n_sel_mtxs=20, device=device, eps=0, eps_min=0)
    # agent.load_model(rootPath)
    # mtx_env = MtxWtsdEnvDN(affs=affs, separating_channel=separating_channel, strides=strides, separating_action=10, offsets=offsets, gt_affinities=affinities)
    action_shape = [4, 56, 56]
    agent = QlAgentUnet(gamma=1, n_state_channels=len(offsets)*2,
                           n_edges=len(offsets), action_shape=action_shape, device=device)
    env = MtxWtsdEnvUnet(affs=affs, separating_channel=separating_channel, offsets=offsets, strides=strides,
                            gt_affinities=affinities, device=device)
    # env.show_current_soln()
    ql = Qlearning(agent, env)
    scores, epss, last_seg = ql.train(n_iterations=5000, batch_size=5, showInterm=False)
    agent.safe_model(rootPath)


if __name__ == '__main__':
    file = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Discs'

    # modelFileSimple = os.path.join(rootPath, 'UnetEdgePredsSimple.pth')
    # dloader = DataLoader(simpleSeg_4_4_Dset(), batch_size=1, shuffle=True, pin_memory=True)
    # affinities_predictor_simple = smallUNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    # affinities_predictor_simple.load_state_dict(torch.load(modelFileSimple), strict=True)
    # affinities_predictor_simple.cuda()

    modelFileCircle = os.path.join(rootPath, 'UnetEdgePreds.pth')
    # trainAffPredCircles(modelFileCircle, device, separating_channel, offsets, strides,)
    dloader = DataLoader(CustomDiscDset(), batch_size=1, shuffle=True, pin_memory=True)
    affinities_predictor_circle = UNet(n_channels=1, n_classes=len(offsets), bilinear=True)
    affinities_predictor_circle.load_state_dict(torch.load(modelFileCircle), strict=True)
    affinities_predictor_circle.cuda()
    #
    q_learning(affinities_predictor_circle, dloader, rootPath)
    # q_testing(affinities_predictor, dloader, rootPath)
    # trainAffPred(modelFile)
    # trainAffPredSimpleImg(modelFile)