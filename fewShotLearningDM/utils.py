import time
import copy
import os
import torch
from fs_model1 import FsModel
import config as cfg
import torch.nn as nn
from torchvision.models.vgg import vgg16_bn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from score import SegmentationMetric
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, phases=['train', 'val'], clusteredFeatures=None):
    model.cuda()
    since = time.time()

    writer = SummaryWriter(logdir=cfg.general.logDir)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(cfg.general.numEpochs):
        print('Epoch {}/{}'.format(epoch, cfg.general.numEpochs - 1))
        print('-' * 10)

        if epoch == 1:
            dummy=1

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            metric = SegmentationMetric(nclass=2)

            with torch.set_grad_enabled(phase == 'train'):
                # backward + optimize only if in training phase
                if phase == 'train':
                        # Iterate over data.
                    if model.trainOnSups == True:
                        for step, (inputs, labels) in enumerate(dataloaders[phase]):
                            supp = inputs
                            supp = supp.to(device)
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            # forward
                            # track history if only in train
                            outputs = model(supp)
                            loss = criterion(outputs)
                            loss.backward()
                            optimizer.step()
                            writer.add_scalar("step/Loss", loss.item(), step+len(dataloaders[phase])*epoch)
                            print('epoch: '+str(epoch)+' step: '+str(step)+' loss: '+str(loss.item()))
                        return clusterFeatures(outputs)
                    else:
                        for step, (inputs, labels) in enumerate(dataloaders[phase]):
                            query = inputs
                            query = query.to(device)
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            # forward
                            # track history if only in train
                            outputs = model(query)
                            loss = criterion(outputs, clusteredFeatures)
                            loss.backward()
                            optimizer.step()
                            writer.add_scalar("step/Loss", loss.item(), step+len(dataloaders[phase])*epoch)
                            print('epoch: '+str(epoch)+' step: '+str(step)+' loss: '+str(loss.item()))


            if phase == 'val':
                # Iterate over data.
                for step, (inputs, labels) in enumerate(dataloaders[phase]):
                    supp, query = inputs
                    supp = supp.to(device)
                    query = query.to(device)
                    labels = labels.to(device)
                    model.trainOnSups = False
                    outputs = model(query)
                    metric.update(outputs, torch.argmax(labels, 1))
                    miou, pixacc = metric.get()
                    writer.add_scalar("step/Accuracy/mIoU", miou, step+len(dataloaders[phase])*epoch)
                    writer.add_scalar("step/Accuracy/pixAcc", pixacc, step+len(dataloaders[phase])*epoch)

        # deep copy the model
        if phase == 'val' and miou > best_acc:
            best_acc = miou
            best_model_wts = copy.deepcopy(model.state_dict())


        # log some images
        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        statInputs, statOutputs, statLabels, statSupports = inputs[1][0:2], preds[0:2], labels[0:2], inputs[0][0:2]

        suppShape = statSupports.shape
        concatSupp = statSupports.view((suppShape[0] * suppShape[1], suppShape[2], suppShape[3], suppShape[4]))

        concatSuppMask = concatSupp[:,1,:,:].unsqueeze(1)
        concatSupp = concatSupp[:, 0, :, :].unsqueeze(1)

        supportGrid1 = vutils.make_grid(concatSuppMask, normalize=True, scale_each=True)
        supportGrid2 = vutils.make_grid(concatSupp, normalize=True, scale_each=True)
        inputGrid = vutils.make_grid(statInputs, normalize=True, scale_each=True)
        outputGrid = vutils.make_grid(statOutputs.unsqueeze(1), normalize=True, scale_each=True)
        labelGrid = vutils.make_grid(torch.argmax(statLabels, 1).unsqueeze(1), normalize=True, scale_each=True)
        writer.add_image('inputs/Image', inputGrid, epoch)
        writer.add_image('outputs/Image', outputGrid, epoch)
        writer.add_image('labels/Image', labelGrid, epoch)
        writer.add_image('supportSets1/Image', supportGrid1, epoch)
        writer.add_image('supportSets2/Image', supportGrid2, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc

def clusterFeatures(features)
    for feat in features:


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
            if idx == 19:
                idx += 4
            else:
                idx += 1

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
