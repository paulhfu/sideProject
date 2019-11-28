import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.densenet import DenseNet

# copied from torchvision/models/densenet.py and edited to fit 3d data

class _FirstLayer(nn.Sequential):
    def __init__(self, num_init_features):
        super(_FirstLayer, self).__init__()
        self.add_module('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
        self.add_module('norm0', nn.BatchNorm3d(num_init_features)),
        self.add_module('relu0', nn.ReLU(inplace=True)),

    def forward(self, x):
        new_features = super(_FirstLayer, self).forward(x)
        return new_features


class _VAE_deconv(nn.Sequential):
    def __init__(self, features, down=True):
        super(_VAE_deconv, self).__init__()
        if down:
            self.add_module('downConv', nn.ConvTranspose3d(features * 2, features,
                                              kernel_size=2, stride=2, bias=False))
        self.add_module('conv1', nn.ConvTranspose3d(features, features,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('norm1', nn.BatchNorm3d(features))

        self.add_module('conv2', nn.ConvTranspose3d(features, features,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('norm2', nn.BatchNorm3d(features))

        self.add_module('conv3', nn.ConvTranspose3d(features, features*2,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('relu3', nn.ReLU(inplace=True))
        self.add_module('norm3', nn.BatchNorm3d(features * 2))


class _VAE_conv(nn.Sequential):
    def __init__(self, features, down=True):
        super(_VAE_conv, self).__init__()
        if down:
            self.add_module('downConv', nn.Conv3d(features, features,
                                              kernel_size=2, stride=2, bias=False))
        self.add_module('conv1', nn.Conv3d(features, features,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('norm1', nn.BatchNorm3d(features))
        self.add_module('relu1', nn.ReLU(inplace=True))

        self.add_module('conv2', nn.Conv3d(features, features,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(features))
        self.add_module('relu2', nn.ReLU(inplace=True))

        self.add_module('conv3', nn.Conv3d(features, features*2,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('norm3', nn.BatchNorm3d(features * 2))
        self.add_module('relu3', nn.ReLU(inplace=True))


class VAE(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):

        super(VAE, self).__init__()

    def forward(self, x):

        return out