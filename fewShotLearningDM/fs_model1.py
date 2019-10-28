import torch
import torch.nn as nn
from collections import OrderedDict
import utils
import config as cfg

class FsModel(nn.Module):

    def __init__(self):
        super(FsModel, self).__init__()
        # self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,], batch_norm=True)
        self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512], batch_norm=True)

    def forward(self, input):
        supSets, img = input[0], input[1]
        num_sups = len(supSets)
        for idx in range(cfg.general.shots):
            supImg = supSets[:, idx, 0:3, :, :]
            supSegCl = supSets[:, idx, 3, :, :]
            supSegBg = supSets[:, idx, 4, :, :]

            f = self.features[0](supImg)
            if idx == 0:
                protFeat0 = torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat0 = torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)
            else:
                protFeat0 += torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat0 += torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)

            f = self.features[1](f)
            if idx == 0:
                protFeat1 = torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat1 = torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)
            else:
                protFeat1 += torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat1 += torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)

            f = self.features[2](f)
            if idx == 0:
                protFeat2 = torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat2 = torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)
            else:
                protFeat2 += torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat2 += torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)

            f = self.features[3](f)
            if idx == 0:
                protFeat3 = torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat3 = torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)
            else:
                protFeat3 += torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
                bgFeat3 += torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)

            # f = self.features[4](f)
            # if idx == 0:
            #     protFeat4 = torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
            #     bgFeat4 = torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)
            # else:
            #     protFeat4 += torch.sum(supSegCl * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3))/torch.sum(supSegCl)
            #     bgFeat4 += torch.sum(supSegBg * nn.functional.interpolate(f, size=img.shape[2:4]), (2, 3)) / torch.sum(supSegBg)

        protFeat = torch.cat([protFeat0/cfg.general.shots,
                             protFeat1/cfg.general.shots,
                             protFeat2/cfg.general.shots,
                             protFeat3/cfg.general.shots], dim = 1)
                             # protFeat4/cfg.general.shots],

        bgFeat = torch.cat([bgFeat0 / cfg.general.shots,
                            bgFeat1 / cfg.general.shots,
                            bgFeat2 / cfg.general.shots,
                            bgFeat3 / cfg.general.shots], dim=1)
                            # bgFeat4 / cfg.general.shots],

        del protFeat0, protFeat1, protFeat2, protFeat3, bgFeat0, bgFeat1, bgFeat2, bgFeat3

        qFeat = torch.empty((protFeat.shape[0], protFeat.shape[1], img.shape[2], img.shape[3])).cuda()
        startDim = 0
        f = self.features[0](img)
        map = nn.functional.interpolate(f, size=img.shape[2:4])
        qFeat[:, startDim:startDim+map.shape[1], :,: ] = map
        startDim += map.shape[1]
        f = self.features[1](f)
        map = nn.functional.interpolate(f, size=img.shape[2:4])
        qFeat[:, startDim:startDim+map.shape[1], :,: ] = map
        startDim += map.shape[1]
        f = self.features[2](f)
        map = nn.functional.interpolate(f, size=img.shape[2:4])
        qFeat[:, startDim:startDim+map.shape[1], :,: ] = map
        startDim += map.shape[1]
        f = self.features[3](f)
        map = nn.functional.interpolate(f, size=img.shape[2:4])
        qFeat[:, startDim:startDim+map.shape[1], :,: ] = map
        startDim += map.shape[1]
        # f = self.features[4](f)
        # map = nn.functional.interpolate(f, size=img.shape[2:4])
        # qFeat[:, startDim:startDim+map.shape[1], :,: ] = map

        protFeat = protFeat.unsqueeze(2)
        protFeat = protFeat.unsqueeze(3)
        protFeat = nn.functional.pad(protFeat, (img.shape[2]-1, 0, img.shape[3]-1, 0), mode='replicate')
        bgFeat = bgFeat.unsqueeze(2)
        bgFeat = bgFeat.unsqueeze(3)
        bgFeat = nn.functional.pad(bgFeat, (img.shape[2]-1, 0, img.shape[3]-1, 0), mode='replicate')

        probProt = nn.functional.cosine_similarity(qFeat, protFeat, dim=1)
        probBg = nn.functional.cosine_similarity(qFeat, bgFeat, dim=1)

        return nn.functional.sigmoid(torch.cat([probProt.unsqueeze(dim=1), probBg.unsqueeze(dim=1)], dim=1))

def getResizedProt(shape, protFeat):
    reseizedProt = torch.empty(shape)
    for h in range(shape[2]):
        for w in range(shape[3]):
            reseizedProt[:, :, h, w] = protFeat
    return reseizedProt