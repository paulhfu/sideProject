import torch
import torch.nn as nn
from collections import OrderedDict
import utils
import config as cfg

class FsModel(nn.Module):

    def __init__(self):
        super(FsModel, self).__init__()
        self.trainOnSups = False
        # self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,], batch_norm=True)
        self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512], batch_norm=True)

    def forward(self, input):
        if self.trainOnSups:
            supSet = input
            supImg = supSet[:, 0:3, :, :]
            downMasks = torch.flatten(supSet[:, 3:5, :, :], start_dim=1, dtype=torch.unint8)

            out, ind = self.features[0](supImg)
            feat0, downMasks = self._maskDownsampledFeatures(out, ind, downMasks)
            out, ind = self.features[1](out)
            feat1, downMasks = self._maskDownsampledFeatures(out, ind, downMasks)
            out, ind = self.features[2](out)
            feat2, downFgMasks = self._maskDownsampledFeatures(out, ind, downMasks)
            out, ind = self.features[3](out)
            feat3, downFgMasks = self._maskDownsampledFeatures(out, ind, downMasks)

            return feat0, feat1, feat2, feat3
        else:
            feat0, ind = self.features[0](input)
            feat1, ind = self.features[1](feat0)
            feat2, ind = self.features[2](feat1)
            feat3, ind = self.features[3](feat2)
            return feat0, feat1, feat2, feat3

    def _maskDownsampledFeatures(self, feat, indi, mask):
        downMasks = torch.zeros((feat.shape[0], 2, feat.shape[2], feat.shape[3]))
        maskedFeat = torch.empty([2]+feat.shape)
        indi = torch.flatten(indi, startDim=2)
        for n in range(feat.shape[0]):
            for c in range(feat.shape[1]):
                downsampledFgMask = torch.index_select(mask[n, 0], 0, indi[n,c,:]).reshape(feat.shape[2:-1])
                downsampledBgMask = torch.index_select(mask[n, 1], 0, indi[n, c, :]).reshape(feat.shape[2:-1])
                maskedFeat[0, n, c, :, :] = torch.masked_select(feat[n, c, :, :], downsampledFgMask)
                maskedFeat[1, n, c, :, :] = torch.masked_select(feat[n, c, :, :], downsampledBgMask)
                downMasks[n, 0, :, :] += downsampledFgMask
                downMasks[n, 1, :, :] += downsampledBgMask
            downMasks[n, :, :, :] /= c
            roundedMasks = torch.empty_as(downMasks, dtype=torch.unint8)
            torch.round(downMasks, roundedMasks)
        return maskedFeat, roundedMasks
