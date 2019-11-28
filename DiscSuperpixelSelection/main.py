import os
from calendar import different_locale

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

import argparse
import config as cfg
from data.datasets import DiscDset
import utils
from torch.utils.data import DataLoader
from models import Policy, V_func
from loss import Policy_loss, V_func_loss
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    args = parser.parse_args()

    files = ['superpixels/superpixels.h5', 'mask/masks.h5']
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Discs'

    dloaders = {'train': DataLoader(DiscDset(rootPath, files, mode='train'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True),
                         'test': DataLoader(DiscDset(rootPath, files, mode='test'), batch_size=cfg.general.trainBatchSize, shuffle=True, pin_memory=True)}

    print('----START TRAINING----' * 4)

    policy = Policy()
    v_func = V_func()

    for param in policy.parameters():
        param.requires_grad = True
    for param in v_func.parameters():
        param.requires_grad = True

    criterion_p = Policy_loss()
    criterion_v = V_func_loss()
    optim_p = torch.optim.SGD(policy.parameters(),  lr=0.001)
    optim_v = torch.optim.SGD(v_func.parameters(), lr=0.001)
    models, val_acc_history, best_acc = utils.train_model(policy, v_func, dloaders, criterion_p, criterion_v, optim_p, optim_v)
    # torch.save(model.state_dict(), os.path.join(cfg.general.checkpointSaveDir, cfg.model.saveToName))
    print('----FINISHED TRAINING----' * 4)
























