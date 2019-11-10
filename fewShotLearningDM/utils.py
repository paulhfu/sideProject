import time
import copy
import os
import torch
from fs_model2 import FsModel
import config as cfg
import kmeans
import torch.nn as nn
from torchvision.models.vgg import vgg16_bn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from score import SegmentationMetric
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getFeaturesAndCompare(model, SLoader, QLoader, criterion, optimizer, clusteredFeatures=None):
    model.cuda()
    since = time.time()

    writer = SummaryWriter(logdir=cfg.general.logDir)
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    metric = SegmentationMetric(nclass=2)

    with torch.set_grad_enabled(False):
        # backward + optimize only if in training phase
        for step, inputs in enumerate(SLoader):
            if not inputs:
                continue
            suppFeats = None
            for supp in inputs:
                supp = nn.functional.pad(supp.unsqueeze(1), (0, 0, 0, 0, 0, 2), mode='replicate').squeeze(0)
                supp = supp.to(device)
            # zero the parameter gradients
            # optimizer.zero_grad()
            # forward
            # track history if only in train
            # loss = criterion(outputs)
                suppFeats = mergeFeats(model(supp), suppFeats)
            # import matplotlib.pyplot as plt;plt.imshow(suppFeats[0][0, 0, :, :].squeeze().detach().cpu().numpy());plt.show()
            # loss.backward()
            # optimizer.step()
            # writer.add_scalar("step/Loss", loss.item(), step)

        for step, (inputs, labels) in enumerate(QLoader):
            query = inputs
            query = query.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            queryFeats = model(query)
            # assigned_features = assignFeatures(outputs, clusteredFeatures)
            # loss = criterion(outputs, clusteredFeatures)
            # loss.backward()
            # optimizer.step()
            # writer.add_scalar("step/Loss", loss.item(), step+len(dataloaders[phase])*epoch)
            # print('epoch: '+str(epoch)+' step: '+str(step)+' loss: '+str(loss.item()))

        dist = getDistance(suppFeats, queryFeats)

def mergeFeats(a, b):
    if b is None:
        return a
    ret = ()
    for f1 ,f2 in zip(a,b):
        ret += ((f1+f2)/2,)
    return a

def getDistance(a, b):
    pass

def clusterFeatures(features):
    ret = []
    for feat in features:
        centers, codes, weights = kmeans.cluster(feat, 500)
        ret.append((centers, weights))
    return ret

def assignClusterToFeatures(clusters, features):
    pass


def make_vgg_blocks(cfg, batch_norm=False):
    layers = []
    vgg_blocks = []
    in_channels = 3
    for idx, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
            vgg_blocks.append(nn.Sequential(*layers))
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            if idx == 0:
                initialize_weights(layers)
            in_channels = v
    vgg_blocks.append(nn.Sequential(*layers))
    return nn.Sequential(*vgg_blocks)

def initialize_weights(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

def load_sdict(fs_model, vgg):
    vgg_model = vgg16_bn()
    vgg_model.load_state_dict(vgg)

    idx = 0
    for subnet in fs_model.features:
        for block in subnet:
            block.load_state_dict(vgg_model.features[idx].state_dict())
            idx += 1
            # if idx == 19:
            #     idx += 4
            # else:
            #     idx += 1

def get_pretrained_model():
    model = FsModel()
    vgg_features = torch.load(os.path.join(cfg.general.checkpointDir, 'vgg_checkpoint.pth'))
    load_sdict(model, vgg_features)

    torch.save(model.state_dict(), os.path.join(cfg.general.checkpointDir, 'fs_model_pretrained_checkpoint.pth'))
    return model

def get_vgg_from_hub():
    vgg = torch.hub.load('pytorch/vision', 'vgg16_bn', pretrained=True)
    torch.save(vgg.state_dict(), os.path.join(cfg.general.checkpointDir, 'vgg_checkpoint.pth'))

def salt_and_pepper(image):
    row, col, ch = image.shape
    sz = row * col * ch
    s_vs_p = 0.5
    amount = cfg.general.noiseLevel / 100
    # Salt mode
    num_salt = np.ceil(amount * sz * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in image.shape]
    image[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * sz * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in image.shape]
    image[coords] = 0
    return image
