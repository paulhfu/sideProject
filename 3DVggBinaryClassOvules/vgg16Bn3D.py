import torch
import torch.nn as nn
from collections import OrderedDict
import utils
import config as cfg
import matplotlib.pyplot as plt
from torchvision.models.vgg import vgg16_bn

class Vgg16Bn3D(nn.Module):

    def __init__(self):
        super(Vgg16Bn3D, self).__init__()
        self.trainOnSups = False
        num_classes = 2
        # self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,], batch_norm=True)
        self.features = utils.make_vgg3D_blocks([64, 'M', 128, 'M', 256, 'M', 512], batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, input):
        feat0, ind = self.features[0](input)
        feat1, ind = self.features[1](feat0)
        feat2, ind = self.features[2](feat1)
        feat3 = self.features[3](feat2)
        # plt.imshow(input[0, 0, :, :].squeeze().detach().cpu().numpy());plt.show()
        x = self.avgpool(feat3)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x