import os
from tensorboardX import SummaryWriter
# os.environ["PYTHONUNBUFFERED"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.insert(0, '/g/kreshuk/hilt/projects/fewShotLearning/mu-net')
from models.ril_function_models import DNDQN
from torch import multiprocessing as mp
from trainers.q_learning import Qlearning
from trainers.reinforce import Reinforce
from trainers.a2c import A2c
from trainers.train_acer import TrainACER
from trainers.train_offpac import TrainOffpac
from trainers.train_offpac_2_models import TrainOffpac2M
import argparse
from agents.tr_opposd_agent_gcn import OPPOSDAgentUnet
from agents.offpac import AgentOffpac
from agents.reinforce_agent_mnm import RIAgentMnm
from agents.a2C_agent_mnm import RIAgentA2c
from environments.sp_grph_gcn_1 import SpGcnEnv
from environments.mtxwtsd_unet import MtxWtsdEnvUnet
from environments.mtxwtsd_mnm import MtxWtsdEnvMNM
import torch
print(torch.__version__)

parser = argparse.ArgumentParser(description='ACER')
## general
parser.add_argument('--num-processes', type=int, default=1, metavar='N', help='Number of training async agents')
parser.add_argument('--n-gpu-per-proc', type=int, default=1, metavar='STEPS', help='Number of gpus per process')
parser.add_argument('--evaluate', action='store_true', help='evaluate existing model')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--target-dir', type=str, default='test1', help='Save folder')
parser.add_argument('--base-dir', type=str, default='/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd', help='Save folder')
## env and model defs
parser.add_argument('--model-name', type=str, default="", metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--n-raw-channels', type=int, default=2, help='number of channels in raw data')
parser.add_argument('--n-embedding-features', type=int, default=128, help='number of embedding feature channels')
parser.add_argument('--n-edge-features', type=int, default=10, help='number of initial edge features')
parser.add_argument('--n-actions', type=int, default=3, help='number of actions on edge')
parser.add_argument('--lstm-hidden-state-size', type=int, default=128, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--reward-function', type=str, default='fully_supervised', help='Reward function')
## warmup feature extractor
parser.add_argument('--fe-extr-warmup', action='store_true', help='pretrain the feature extractor with contrastive loss')
parser.add_argument('--fe-warmup-iterations', type=int, default=100, metavar='SIZE', help='number of iterations of feature extrqactor warmup')
parser.add_argument('--fe-warmup-batch-size', type=int, default=10, metavar='SIZE', help='batch size for feature extractor warmup')
## main training (env, trainer)
parser.add_argument('--T-max', type=int, default=100, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=3, metavar='STEPS', help='Max number of forward steps before update')
parser.add_argument('--max-episode-length', type=int, default=10, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--init-epsilon', type=float, default=1, metavar='eps', help='initial epsilon value for exploration')
parser.add_argument('--eps-final', type=float, default=0.05, metavar='eps', help='final epsilon')
parser.add_argument('--action-agression', type=float, default=0.2, help='value by which one action changes state')
parser.add_argument('--eps-rule', type=str, default='gaussian', help='epsilon rule')
parser.add_argument('--stop-qual-rule', type=str, default='gaussian', help='epsilon rule')
parser.add_argument('--stop-qual-final', type=float, default=0.001, metavar='eps', help='final epsilon')
parser.add_argument('--stop-qual-scaling', type=float, default=70, metavar='eps', help='final epsilon')
parser.add_argument('--stop-qual-offset', type=float, default=30, metavar='eps', help='final epsilon')
## Training specifics (agent)
# parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trust-region', action='store_true', help='use trust region gradient')
parser.add_argument('--discount', type=float, default=0.5, metavar='γ', help='Discount factor')  # for acer this is 0.99
parser.add_argument('--lbd', type=float, default=1, metavar='lambda', help='lambda elegibility trace parameter')
parser.add_argument('--qnext-replace-cnt', type=int, default=5, help='number of learning steps after which qnext is updated')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight averaging rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--trust-region-weight', type=float, default=.5, metavar='lbd', help='Trust region regularization weight')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='entropy', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=1000, metavar='VALUE', help='Gradient L2 norm clipping')
parser.add_argument('--weight-l2-reg-params-weight', type=float, default=0, metavar='VALUE', help='Gradient L2 weight')
## Optimization
parser.add_argument('--min-lr', type=float, default=0.001, metavar='η', help='min Learning rate')
parser.add_argument('--lr', type=float, default=0.001, metavar='η', help='Learning rate')
parser.add_argument('--Adam-weight-decay', type=float, default=0, metavar='wdec', help='Adam weight decay')
parser.add_argument('--Adam-betas', type=float, default=[0.9, 0.999], metavar='β', help='Adam decay factors')

if __name__ == '__main__':
    # test_model()
    # sp_fe_ae.main()
    # exit()
    # file = 'mask/masks.h5'
    # rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/models'
    #
    # modelFileSimple = os.path.join(rootPath, 'UnetEdgePredsSimple.pth')
    # dloader = DataLoader(simpleSeg_4_4_Dset(), batch_size=1, shuffle=True, pin_memory=True)
    # affinities_predictor_simple = smallUNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    # affinities_predictor_simple.load_state_dict(torch.load(modelFileSimple), strict=True)
    # affinities_predictor_simple.cuda()

    # modelFileCircle = os.path.join(rootPath, 'UnetEdgePreds.pth')
    # modelFileCircleG1 = os.path.join(rootPath, 'UnetEdgePredsG1.pth')
    # trainAffPredCircles(modelFileCircle, device, separating_channel, offsets, strides,)
    # a=1
    # affinities_predictor_circle = UNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    # affinities_predictor_circle.load_state_dict(torch.load(modelFileCircle), strict=True)
    # affinities_predictor_circle.cuda()
    # dloader_disc = DataLoader(CustomDiscDset(affinities_predictor_circle, separating_channel), batch_size=1, shuffle=True,
    #                      pin_memory=True)
    # dloader_simple_img = DataLoader(SimpleSeg_20_20_Dset(), batch_size=1, shuffle=True,
    #                      pin_memory=True)
    #
    # dloader_discs = DataLoader(DiscSpGraphDset(affinities_predictor_circle, separating_channel, offsets), batch_size=1,
    #                           shuffle=True, pin_memory=True)
    # for i, a in enumerate(dloader_discs):
    #     b=1


    # q_learning(dloader_mult_discs, rootPath, args, learn=True)
    print('visible gpus: ', torch.cuda.device_count())
    args = parser.parse_args()
    mp.set_start_method('spawn')
    trainer = TrainOffpac(args)
    scores = trainer.train()
