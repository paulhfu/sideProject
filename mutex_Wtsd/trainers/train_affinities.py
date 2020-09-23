import torch
from mutex_watershed import compute_mws_segmentation_cstm
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation.mws import get_valid_edges
from models.simple_unet import UNet, smallUNet
from data.disc_datasets import simpleSeg_4_4_Dset, CustomDiscDset
from torch.utils.data import DataLoader
import time
import torch
import torch.nn as nn
from mutex_watershed import compute_partial_mws_prim_segmentation
import numpy as np


def trainAffPredSimpleImg(saveToFile, device, separating_channel, offsets, strides, numEpochs=2):
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

    return model


def trainAffPredCircles(saveToFile, device, separating_channel, offsets, strides, numEpochs=8):
    file = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/data/Discs'

    dloader = DataLoader(CustomDiscDset(length=5), batch_size=1, shuffle=True, pin_memory=True)
    print('----START TRAINING----' * 4)

    model = UNet(n_channels=1, n_classes=len(offsets), bilinear=True)
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters())

    model.cuda()
    since = time.time()

    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Iterate over data.
        for step, (inputs, _, affinities) in enumerate(dloader):
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
        # weights[separating_channel:] /= 2
        affs = affinities.squeeze().detach().cpu().numpy()
        weights[separating_channel:] *= -1
        weights[separating_channel:] += +1
        affs[separating_channel:] *= -1
        affs[separating_channel:] += +1

        weights[:separating_channel] /= 1.5

        ndim = len(offsets[0])
        assert all(len(off) == ndim for off in offsets)
        image_shape = weights.shape[1:]
        valid_edges = get_valid_edges(weights.shape, offsets, separating_channel, strides, False)
        node_labeling1, cut_edges, used_mtxs, neighbors_features = compute_partial_mws_prim_segmentation(weights.ravel(),
                                                      valid_edges.ravel(),
                                                      offsets,
                                                      separating_channel,
                                                      image_shape)
        node_labeling_gt = compute_mws_segmentation(affs, offsets, separating_channel, algorithm='kruskal')
        labels = compute_mws_segmentation(weights, offsets, separating_channel, algorithm='kruskal')
        # labels, neighbors, cutting_edges, mutexes = compute_mws_segmentation_cstm(weights, offsets, separating_channel)
        edges = np.zeros(affs.shape).ravel()
        # lbl = 1
        # for cut_edges, rep_edges in zip(cutting_edges, mutexes):
        #     for edge in cutting_edges:
        #         edges[edge] = lbl
        #     for edge in rep_edges:
        #         edges[edge] = lbl
        #     lbl += 1
        # edges = edges.reshape(affs.shape)
        import matplotlib.pyplot as plt
        from matplotlib import cm
        labels = labels.reshape(image_shape)
        labels1 = node_labeling1.reshape(image_shape)
        node_labeling_gt = node_labeling_gt.reshape(image_shape)

        # show_edge1 = cm.prism(edges[0] / edges[0].max())
        # show_edge2 = cm.prism(edges[1] / edges[1].max())
        # show_edge3 = cm.prism(edges[2] / edges[2].max())
        # show_edge4 = cm.prism(edges[3] / edges[3].max())

        show_seg1 = cm.prism(labels1 / labels1.max())
        show_seg = cm.prism(labels / labels.max())
        show_seg2 = cm.prism(node_labeling_gt / node_labeling_gt.max())
        show_raw = cm.gray(inputs.squeeze().detach().cpu().numpy())
        # img1 = np.concatenate([np.concatenate([show_edge1, show_edge2], axis=1),
        #                       np.concatenate([show_edge3, show_edge4], axis=1)], axis=0)
        img2 = np.concatenate([np.concatenate([show_seg, show_seg1], axis=1),
                              np.concatenate([show_raw, show_seg2], axis=1)], axis=0)
        # plt.imshow(img1); plt.show()
        # plt.imshow(img2); plt.show()

    torch.save(model.state_dict(), saveToFile)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return

