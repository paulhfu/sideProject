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

    if cfg.general.augmentImgs:
        # setup transforms in order to have all images in the dataset equally sized which is required for larger batch sizes
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1)),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * cfg.general.noiseLevel),
            transforms.Lambda(utils.salt_and_pepper),
        ])
        supp_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * cfg.general.noiseLevel),
            transforms.Lambda(utils.salt_and_pepper),
        ])
        tgt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0].round().long()),  # for some reason the loss function wants the target type to be long
        ])
    else:
        # setup transforms in order to have all images in the dataset equally sized which is required for larger batch sizes
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1)),
        ])
        supp_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
        tgt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0].round().long()),  # for some reason the loss function wants the target type to be long
        ])

    print('----START TRAINING----' * 4)
    # accs = {}
    # utils.get_pretrained_model()
    # model = FsModel()
    model = utils.get_pretrained_model()
    # model.load_state_dict(torch.load(os.path.join(cfg.general.checkpointDir, cfg.model.loadFromName)), strict=True)

    for param in model.parameters():
        param.requires_grad = True
    if cfg.model.freezeVgg:
        for param in model.features.parameters():
            param.requires_grad = False

    criterion = HyperplaneDistLoss(weights=torch.tensor(cfg.general.lossWeights, device=device))
    optimizer = torch.optim.Adam(model.parameters())

    for dataIdx in cfg.general.dataIdx:
        if cfg.general.usePascalVOC:
            train_dset = loadDataSet.Pascal5FewShotDset(os.getcwd(), "train", dataIdx, transform=transform, tgt_transform=tgt_transform, supp_transform=supp_transform,
                                                        shuffle_lasses=cfg.general.shuffleClasses, shots=cfg.general.shots)
            test_dset = loadDataSet.Pascal5FewShotDset(os.getcwd(), "test", dataIdx, transform=transform, tgt_transform=tgt_transform, supp_transform=supp_transform,
                                                       shuffle_lasses=cfg.general.shuffleClasses, shots=cfg.general.shots)

            train_loader = DataLoader(train_dset, batch_size=cfg.general.trainBatchSize, shuffle=False, pin_memory=True)

            validate_loader = DataLoader(test_dset, batch_size=cfg.general.testBatchSize, shuffle=False, pin_memory=True)

            dataloaders = {'train' : train_loader, 'val' : validate_loader}
        else:
            train_dset = randObjDset.RandObjDset(os.getcwd(), transform=transform, tgt_transform=tgt_transform,
                                                 supp_transform=supp_transform, shots=cfg.general.shots)

            train_loader = DataLoader(train_dset, batch_size=cfg.general.trainBatchSize,
                                      shuffle=cfg.general.shuffleClasses, pin_memory=True)

            dataloaders = {'train': train_loader}

        if cfg.general.usePascalVOC:
            model, val_acc_history, best_acc = utils.train_model(model, dataloaders, criterion, optimizer,
                                                                 phases=['train', 'val'])
        else:
            model, val_acc_history, best_acc = utils.train_model(model, dataloaders, criterion, optimizer,
                                                                 phases=['train'])
        # accs.append(best_acc)
    torch.save(model.state_dict(), os.path.join(cfg.general.checkpointSaveDir, cfg.model.saveToName))
    print('----FINISHED TRAINING----' * 4)

# import matplotlib.pyplot as plt; plt.imshow(q_o.permute(1,2,0));plt.show()
























