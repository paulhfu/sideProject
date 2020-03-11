import os
from tensorboardX import SummaryWriter
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
from agents.tr_opposd_agent_gcn import OPPOSDAgentUnet
from agents.ql_agent_unet import QlAgentUnet
from agents.reinforce_agent_unet import RIAgentUnet
from agents.qlretrace_agent_unet import QlRetraceAgentUnet
from agents.reinforce_agent_mnm import RIAgentMnm
from agents.a2C_agent_mnm import RIAgentA2c
from agents.ql_gcn_agent_1 import QlAgentGcn1
from environments.sp_grph_gcn_1 import SpGcnEnv
from environments.mtxwtsd_unet_fcn import MtxWtsdEnvUnetFcn
from environments.mtxwtsd_unet import MtxWtsdEnvUnet
from environments.mtxwtsd_mnm import MtxWtsdEnvMNM
from models.simple_unet import UNet, smallUNet
from data.datasets import simpleSeg_4_4_Dset, CustomDiscDset, SimpleSeg_20_20_Dset, DiscSpGraphDset
from torch.utils.data import DataLoader
import time
import torch
from models import sp_fe_ae
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

def q_learning(dloader, rootPath, learn=True):
    # raw, affinities, gt_affinities = next(iter(dloader))
    # affinities = affinities.squeeze().detach().cpu().numpy()
    # gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
    # action_shape = [4, 56, 56]
    writer = SummaryWriter(logdir='./logs')
    # agent = QlAgentMNM(gamma=1, n_state_channels=2, n_actions=2, device=device, eps=1)
    env = SpGcnEnv(writer=writer, device=device)
    # agent = QlAgentGcn1(gamma=0.5, lambdA=1, n_actions=3, device=device, env=env, epsilon=1, mem_size=10, train_phase=1, writer=writer)
    agent = OPPOSDAgentUnet(gamma=0.5, lambdA=1, n_actions=3, device=device, env=env, epsilon=1, mem_size=10, train_phase=1, writer=writer)
    agent.load_model(rootPath)
    # env.execute_opt_policy()
    # env.show_current_soln()
    # agent = QlAgentUnet(gamma=1, n_state_channels=len(offsets)*2,
    #                     n_edges=len(offsets), n_actions=3, action_shape=action_shape, device=device)
    # env = MtxWtsdEnvUnet(affinities=affinities, separating_channel=separating_channel, offsets=offsets, strides=strides,
    #                      gt_affinities=gt_affinities)
    # n_iterations = 1000
    # bs = 5
    ql = Qlearning(agent, env, dloader=dloader)
    if learn:
        # agent.load_model(rootPath)
        scores, epss, last_seg = ql.train_retrace_gcn(n_iterations=1000, limiting_behav_iter=10000)
        agent.safe_model(rootPath)
    else:
        agent.load_model(rootPath)
        env.show_current_soln()
        soln = ql.test()
        env.show_current_soln()

def reinforce(dloader, rootPath, learn=True):
    raw, affinities, gt_affinities = next(iter(dloader))
    affinities = affinities.squeeze().detach().cpu().numpy()
    gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
    action_shape = [1]
    agent = RIAgentMnm(gamma=1, n_state_channels=2,
                       n_actions=2, action_shape=action_shape, device=device)
    env = MtxWtsdEnvMNM(affinities=affinities, separating_channel=separating_channel, offsets=offsets, strides=strides,
                        gt_affinities=gt_affinities, stop_cnt=200, win_reward=-180)
    # agent = RIAgentUnet(gamma=1, n_state_channels=len(offsets)*2,
    #                     n_edges=len(offsets), n_actions=3, action_shape=action_shape, device=device)
    agent.load_model(directory=rootPath)
    # env = MtxWtsdEnvUnet(affinities=affinities, separating_channel=separating_channel, offsets=offsets, strides=strides,
    #                      gt_affinities=gt_affinities, keep_first_state=False, stop_cnt=15)
    n_iterations = 500
    ri = Reinforce(agent, env, dloader)
    if learn:
        scores, steps, last_seg = ri.train(n_iterations=n_iterations, showInterm=False)
        agent.safe_model(rootPath)
    else:
        env.show_current_soln()
        soln = ri.test()
        env.show_current_soln()

def a2c(dloader, rootPath, learn=True):
    raw, affinities, gt_affinities = next(iter(dloader))
    affinities = affinities.squeeze().detach().cpu().numpy()
    gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
    action_shape = [1]
    agent = RIAgentA2c(gamma=1, n_state_channels=2,
                       n_actions=2, action_shape=action_shape, device=device, epsilon=1)
    # agent.load_model(rootPath)
    env = MtxWtsdEnvMNM(affinities=affinities, separating_channel=separating_channel, offsets=offsets, strides=strides,
                        gt_affinities=gt_affinities, stop_cnt=15, win_reward=-9, only_prop_improv=False)
    # env.execute_opt_policy()

    n_iterations = 1000  # 4000
    a2c = A2c(agent, env, dloader)
    if learn:
        scores, steps, last_seg = a2c.train(n_iterations=n_iterations, showInterm=False)
        agent.safe_model(rootPath)
    else:
        agent.load_model(rootPath)
        env.show_current_soln()
        soln = a2c.test()
        env.show_current_soln()

def opposd(dloader, rootPath, learn=True):
    raw, affinities, gt_affinities = next(iter(dloader))
    affinities = affinities.squeeze().detach().cpu().numpy()
    gt_affinities = gt_affinities.squeeze().detach().cpu().numpy()
    action_shape = [1]
    agent = OPPOSDAgentUnet(gamma=1, n_state_channels=2,
                       n_actions=2, action_shape=action_shape, device=device, epsilon=1)
    # agent.load_model(rootPath)
    env = MtxWtsdEnvUnet(affinities=affinities, separating_channel=separating_channel, offsets=offsets, strides=strides,
                        gt_affinities=gt_affinities, stop_cnt=15, win_reward=-9, only_prop_improv=False)
    # env.execute_opt_policy()

    n_iterations = 1000  # 4000
    a2c = A2c(agent, env, dloader)
    if learn:
        scores, steps, last_seg = a2c.train(n_iterations=n_iterations, showInterm=False)
        agent.safe_model(rootPath)
    else:
        agent.load_model(rootPath)
        env.show_current_soln()
        soln = a2c.test()
        env.show_current_soln()

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
    sp_fe_ae.main()
    exit()
    file = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/models'
    #
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
    dloader_disc = DataLoader(DiscSpGraphDset(affinities_predictor_circle, separating_channel, offsets), batch_size=1,
                              shuffle=True, pin_memory=True)
    q_learning(dloader_disc, rootPath, learn=True)
    # reinforce(dloader_disc, rootPath, learn=True)
    # a2c(dloader_simple_img, rootPath, learn=False)

