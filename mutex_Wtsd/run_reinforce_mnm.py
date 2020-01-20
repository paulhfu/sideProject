import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from models.ril_function_models import DNDQN, UnetFcnDQN, UnetDQN
from train_affinities import trainAffPredCircles, trainAffPredSimpleImg
from mutex_watershed import compute_mws_segmentation_cstm
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation.mws import get_valid_edges
from q_learning import Qlearning
from reinforce import Reinforce
from a2c import A2c
from agents.ql_agent_unet_fcn import QlAgentUnetFcn
from agents.ql_agent_dn import QlAgentDN
from agents.ql_agent_mnm import QlAgentMNM
from agents.ql_agent_unet import QlAgentUnet
from agents.reinforce_agent_unet import RIAgentUnet
from agents.reinforce_agent_mnm import RIAgentMnm
from agents.a2C_agent_mnm import RIAgentA2c
from environments.mtxwtsd_unet_fcn import MtxWtsdEnvUnetFcn
from environments.mtxwtsd_unet import MtxWtsdEnvUnet
from environments.mtxwtsd_mnm import MtxWtsdEnvMNM
from models.simple_unet import UNet, smallUNet
from data.datasets import simpleSeg_4_4_Dset, CustomDiscDset, SimpleSeg_20_20_Dset
from torch.utils.data import DataLoader
from main import reinforce
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


def test_model():
    q_eval = DNDQN(num_classes=2, num_inchannels=2, device=device, block_config=(6,))
    tgt_finQ = torch.tensor([0, 0], dtype=torch.float32, device=q_eval.device)
    q_eval.cuda(device=device)
    while True:
        finQ = q_eval(torch.ones((1, 2, 40, 40), device=q_eval.device))

        loss = q_eval.loss(tgt_finQ, finQ)
        print(loss.item())
        q_eval.optimizer.zero_grad()
        loss.backward()
        # for param in self.q_eval.parameters():
        #     param.grad.data.clamp_(-1, 1)
        q_eval.optimizer.step()


if __name__ == '__main__':
    # test_model()
    file = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/models'

    # modelFileSimple = os.path.join(rootPath, 'UnetEdgePredsSimple.pth')
    # dloader = DataLoader(simpleSeg_4_4_Dset(), batch_size=1, shuffle=True, pin_memory=True)
    # affinities_predictor_simple = smallUNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    # affinities_predictor_simple.load_state_dict(torch.load(modelFileSimple), strict=True)
    # affinities_predictor_simple.cuda()

    modelFileCircle = os.path.join(rootPath, 'UnetEdgePreds.pth')
    modelFileCircleG1 = os.path.join(rootPath, 'UnetEdgePredsG1.pth')
    # trainAffPredCircles(modelFileCircle, device, separating_channel, offsets, strides,)
    a=1
    affinities_predictor_circle = UNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    affinities_predictor_circle.load_state_dict(torch.load(modelFileCircle), strict=True)
    affinities_predictor_circle.cuda()
    dloader_disc = DataLoader(CustomDiscDset(affinities_predictor_circle, separating_channel), batch_size=1, shuffle=True,
                         pin_memory=True)
    dloader_simple_img = DataLoader(SimpleSeg_20_20_Dset(), batch_size=1, shuffle=True,
                         pin_memory=True)
    #
    reinforce(dloader_simple_img, rootPath, learn=True)

