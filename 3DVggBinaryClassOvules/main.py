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
from data.datasets import OvuleDset, TomatoeDset, createOvuleData, splitTrainTestSets
import utils
from torch.utils.data import DataLoader
from vgg16Bn3D import Vgg16Bn3D
from denseNet121_3D import DenseNet3D
import h5py
import os
import numpy as np
from loss import DiceLoss, CeLoss, HyperplaneDistLoss, HypercubeDistLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    args = parser.parse_args()

    files = ['N_491.h5', 'N_226_ds2x.h5', 'N_226_ds3x.h5', 'N_290.h5', 'N_290_ds2x.h5', 'N_290_ds3x.h5']
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Ovules'


    # files = [h5py.File(os.path.join(rootPath, 'N_482_ds2x_Cells.h5'), 'a'), h5py.File(os.path.join(rootPath, 'N_482_ds2x_McSeg_Cells.h5'), 'a')]
    # for file in files:
    #     keys = list(file.keys())
    #     for key in keys:
    #         if np.any(file[key][:].shape<16) or np.linalg.norm(file[key][:].shape) > 120:
    #             del file[key]
    #     file.close()
    #

    sourceFiles1 = ['N_482_ds2x.h5', 'N_482_ds2x_McSeg.h5']
    tgtFiles1 = ['N_482_ds2x_Cells.h5', 'N_482_ds2x_McSeg_Cells.h5']
    sourceFiles2 = ['batch_N394.h5', 'batch_N394_McSeg.h5']
    tgtFiles2 = ['batch_N394_Cells.h5', 'batch_N394_McSeg_Cells.h5']
    sourceFiles3 = ['/g/kreshuk/wolny/Datasets/Ovules/train/N_491.h5', os.path.join(rootPath, 'N_491_McSeg.h5')]
    tgtFiles3 = ['N_491_Cells.h5', 'N_491_McSeg_Cells.h5']

    # # createOvuleData("", sourceFiles3, tgtFiles3)
    # ovule_dset1 = OvuleDset(rootPath, tgtFiles1)
    # ovule_dset2 = OvuleDset(rootPath, tgtFiles2)
    # ovule_dset3 = OvuleDset(rootPath, tgtFiles3)
    #
    # # tomato_dset = TomatoeDset('/g/kreshuk/hilt/projects/fewShotLearning/data/RichardTomatoMeristem', ['meristem_T0_PI.h5'])
    #
    # # files = ['N_536.h5','N_536_ds2x.h5','N_536.h5','N_563_ds2x.h5','N_563_ds3x.h5', 'N_226.h5', 'N_226_ds2x.h5', 'N_226_ds3x.h5', 'N_290.h5', 'N_290_ds2x.h5', 'N_290_ds3x.h5']
    # # train_dset = OvulesDset('/g/kreshuk/wolny/Datasets/Ovules/train', files, train=True, shuffle=cfg.general.shuffleClasses)
    #
    # ovule_loader1 = DataLoader(ovule_dset1, batch_size=cfg.general.trainBatchSize, shuffle=False, pin_memory=True)
    # ovule_loader2 = DataLoader(ovule_dset2, batch_size=cfg.general.trainBatchSize, shuffle=False, pin_memory=True)
    # ovule_loader3 = DataLoader(ovule_dset3, batch_size=cfg.general.trainBatchSize, shuffle=False, pin_memory=True)
    dataloaders_dset1 = {'train': DataLoader(OvuleDset(rootPath, tgtFiles1, mode='train'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True),
                         'test': DataLoader(OvuleDset(rootPath, tgtFiles1, mode='test'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True)}
    dataloaders_dset2 = {'train': DataLoader(OvuleDset(rootPath, tgtFiles2, mode='train'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True),
                         'test': DataLoader(OvuleDset(rootPath, tgtFiles2, mode='test'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True)}
    dataloaders_dset3 = {'train': DataLoader(OvuleDset(rootPath, tgtFiles3, mode='train'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True),
                         'test': DataLoader(OvuleDset(rootPath, tgtFiles3, mode='test'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True)}

    # tomato_loader = DataLoader(tomato_dset, batch_size=cfg.general.testBatchSize, shuffle=False, pin_memory=True)

    model = DenseNet3D()

    # for param in model.parameters():
    #     param.requires_grad = True
    #
    # print('----START TRAINING----' * 4)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(),  lr=0.001)
    # model, val_acc_history, best_acc = utils.train_model(model, (dataloaders_dset1, dataloaders_dset2, dataloaders_dset3), criterion, optimizer)
    # torch.save(model.state_dict(), os.path.join(cfg.general.checkpointSaveDir, cfg.model.saveToName))
    # print('----FINISHED TRAINING----' * 4)



    model.load_state_dict(torch.load(os.path.join(cfg.general.checkpointDir, cfg.model.loadFromName)), strict=True)

    for param in model.parameters():
        param.requires_grad = False

    utils.do_inference(model, (dataloaders_dset1, dataloaders_dset2, dataloaders_dset3))



















