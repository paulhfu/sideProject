import time
import copy
import os
import torch
from models import policy, v_func
import config as cfg
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from score import ClassificationMetric
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, phases=['train', 'test']):
    model.cuda()
    since = time.time()

    writer = SummaryWriter(logdir=cfg.general.logDir)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_loss = 0.0
    tainStep = 0
    testStep = 0
    bn = torch.nn.BatchNorm3d(1, affine=False, track_running_stats=False)

    for epoch in range(cfg.general.numEpochs):
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for idx, dataloader in enumerate(dataloaders):
                print('Epoch {}/{}'.format(epoch, cfg.general.numEpochs - 1))
                print('-' * 10)
                metric = ClassificationMetric(nclass=2)
                # Iterate over data.
                for input, label in dataloader[phase]:
                    if np.sum(input.shape[2:len(input.shape)]) > 300:
                        continue
                    input = bn(input)
                    input = input.to(device)
                    label = label.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = model(input)
                        loss = criterion(output, label)
                        preds = torch.argmax(output, 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            writer.add_scalar("step/TrainLoss", loss.item(), tainStep)
                            print('epoch: '+str(epoch)+' step: '+str(tainStep))
                            tainStep += 1
                            running_loss += loss.item() * label.shape[0]
                            writer.add_scalar("step/running_loss", running_loss, tainStep)
                        else:
                            metric.update(output, label)
                            m = metric.get()
                            writer.add_scalar("step/Accuracy/l2dist", m, testStep)
                            writer.add_scalar("step/TestLoss", loss.item(), testStep)
                            testStep += 1

            # deep copy the model
            if phase == 'test' and m < best_acc:
                best_acc = metric
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc

def mergeFeats(a, b):
    if b is None:
        return a
    ret = ()
    for f1, f2 in zip(a,b):
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

def make_vgg3D_blocks(cfg, batch_norm=False):
    layers = []
    vgg_blocks = []
    in_channels = 1
    for idx, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)]
            vgg_blocks.append(nn.Sequential(*layers))
            layers = []
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            initialize_weights(layers)
            in_channels = v
    vgg_blocks.append(nn.Sequential(*layers))
    return nn.Sequential(*vgg_blocks)

def initialize_weights(modules):
    for module in modules:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

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
