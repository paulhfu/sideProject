import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

import argparse
import config as cfg
from pascal5 import loadDataSet
from data.datasets import OvuleDset, TomatoeDset
from pascal5 import randObjDset
from torchvision import models
import utils
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fs_model1 import FsModel
from loss import DiceLoss, CeLoss, HyperplaneDistLoss, HypercubeDistLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    args = parser.parse_args()

    files = ['N_491.h5', 'N_226_ds2x.h5', 'N_226_ds3x.h5', 'N_290.h5', 'N_290_ds2x.h5', 'N_290_ds3x.h5']
    ovule_dset = OvuleDset('/g/kreshuk/wolny/Datasets/Ovules/train', ['N_491.h5', ''])
    tomato_dset = TomatoeDset('/g/kreshuk/hilt/projects/fewShotLearning/data/RichardTomatoMeristem', ['meristem_T0_PI.h5'])



    # files = ['N_536.h5','N_536_ds2x.h5','N_536.h5','N_563_ds2x.h5','N_563_ds3x.h5', 'N_226.h5', 'N_226_ds2x.h5', 'N_226_ds3x.h5', 'N_290.h5', 'N_290_ds2x.h5', 'N_290_ds3x.h5']
    # train_dset = OvulesDset('/g/kreshuk/wolny/Datasets/Ovules/train', files, train=True, shuffle=cfg.general.shuffleClasses)

    # for data in train_dset:
    #     pass

    ovule_loader = DataLoader(ovule_dset, batch_size=cfg.general.trainBatchSize, shuffle=False, pin_memory=True)

    tomato_loader = DataLoader(tomato_dset, batch_size=cfg.general.testBatchSize, shuffle=False, pin_memory=True)

    print('----START TRAINING----' * 4)
    # accs = {}
    # utils.get_pretrained_model()
    # model = FsModel()
    model = utils.get_pretrained_model()
    # model.load_state_dict(torch.load(os.path.join(cfg.general.checkpointDir, cfg.model.loadFromName)), strict=True)

    for param in model.parameters():
        param.requires_grad = True
    # if cfg.model.freezeVgg:
    #     for param in model.features.parameters():
    #         param.requires_grad = False


    criterion = HypercubeDistLoss(weights=torch.tensor(cfg.general.lossWeights, device=device))
    optimizer = torch.optim.SGD(model.parameters(),  lr=0.001)

    model, val_acc_history, best_acc = utils.getFeaturesAndCompare(model, ovule_loader, tomato_loader, criterion, optimizer)

    # accs.append(best_acc)
    torch.save(model.state_dict(), os.path.join(cfg.general.checkpointSaveDir, cfg.model.saveToName))
    print('----FINISHED TRAINING----' * 4)

# import matplotlib.pyplot as plt; plt.imshow(q_o.permute(1,2,0));plt.show()
























