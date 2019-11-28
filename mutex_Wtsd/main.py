import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from affogato.segmentation.mws import get_valid_edges
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation.mws import get_valid_edges
from simple_unet import UNet
from datasets import DiscDset, computeAffs
from torch.utils.data import DataLoader
import time
import copy
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
from mutex_watershed import compute_mws_prim_segmentation

import h5py
import os
import numpy as np

# offsets = [[-1, 0], [0, -1], [-1, -1], [1, -1],
#            # direct 3d nhood for attractive edges
#            [-9, 0], [0, -9], [-9, -9], [9, -9],
#            # inplane diagonal dam edges
#            [-15, 0], [0, -15], [-15, -15], [15, -15]]
strides = [4, 4]
separating_channel = 2
offsets = [[-1, 0], [0, -1],
           # direct 3d nhood for attractive edges
           [-9, 0], [0, -9]]
           # inplane diagonal dam edges
           # [-15, 0], [0, -15]]

def q_learning(modelFile):
    file_raw = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Discs'
    dloader = DataLoader(DiscDset(rootPath, file, mode='train'), batch_size=1, shuffle=True, pin_memory=True)

    number_actions = 3
    number_mtx_offs = len(offsets) - separating_channel
    Q = UNet(n_channels=number_mtx_offs + 1, n_classes=number_actions*number_mtx_offs, bilinear=True)

    affinitie_predictor = UNet(n_channels=1, n_classes=len(offsets), bilinear=True)
    affinitie_predictor.load_state_dict(torch.load(modelFile), strict=True)
    affinitie_predictor.cuda()
    raw, affinities = next(iter(dloader)
    with torch.set_grad_enabled(False):
        pred_affinities = affinitie_predictor(raw)


def trainAffPred(saveToFile, numEpochs=3):
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
    modelFile = os.path.join(rootPath, 'UnetEdgePreds.pth')
    # computeAffs(os.path.join(rootPath, files), offsets)
    # q_learning(modelFile)
    trainAffPred(modelFile)