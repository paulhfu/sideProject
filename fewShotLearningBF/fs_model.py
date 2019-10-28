import torch
import torch.nn as nn
from collections import OrderedDict
import utils
import config as cfg

class FsModel_vgg_unet1(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, init_features=32):
        super(FsModel_vgg_unet1, self).__init__()

        self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'], batch_norm=True)
        self.inter_cons = utils.make_ia_blocks([64, 128, 256, 512], batch_norm=True)

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = _block(features * 2, features, name="dec1")

        self.lcf_conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, input):
        supSets, img = input[0], input[1]
        enc1 = self.encoder1(img)
        e2 = self.encoder2(self.pool1(enc1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        for idx in range(cfg.general.shots):
            supSet = supSets[:,idx,:,:,:]

            f1 = self.features[0](supSet)
            if idx == 0: enc2 = self.inter_cons[0](torch.cat((f1, e2), 1))
            else: enc2 = torch.add(enc2, self.inter_cons[0](torch.cat((f1, e2), 1)))

            f2 = self.features[1](f1)
            if idx == 0: enc3 = self.inter_cons[1](torch.cat((f2, e3), 1))
            else: enc3 = torch.add(enc3, self.inter_cons[1](torch.cat((f2, e3), 1)))

            f3 = self.features[2](f2)
            if idx == 0: enc4 = self.inter_cons[2](torch.cat((f3, e4), 1))
            else: enc4 = torch.add(enc4, self.inter_cons[2](torch.cat((f3, e4), 1)))

            f4 = self.features[3](f3)
            if idx == 0: bottleneck = self.inter_cons[3](torch.cat((f4, bn), 1))
            else: bottleneck = torch.add(bottleneck, self.inter_cons[3](torch.cat((f4, bn), 1)))

        enc2 = torch.div(enc2, cfg.general.shots)
        enc3 = torch.div(enc3, cfg.general.shots)
        enc4 = torch.div(enc4, cfg.general.shots)
        bottleneck = torch.div(bottleneck, cfg.general.shots)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.lcf_conv(dec1))


class FsModel_vgg_unet2(nn.Module):

    def __init__(self, in_channels=32, out_channels=2):
        super(FsModel_vgg_unet2, self).__init__()
                                            # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = utils.make_vgg_blocks([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512], batch_norm=True)
        self.inter_cons = utils.make_uia_blocks([64, 128, 256, 512, 512])

        features = in_channels
        self.encoder1 = _block(3, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = _block(features * 2, features, name="dec1")

        self.lcf_conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, input):
        supSets, img = input[0], input[1]

        enc1 = self.encoder1(img)
        e2 = self.encoder2(self.pool1(enc1))

        for idx in range(cfg.general.shots):
            supSet = supSets[:,idx,:,:,:]
            f1 = self.features[0](supSet)
            e2 = self.inter_cons[0](torch.cat((f1, e2), 1))

            # import matplotlib.pyplot as plt; plt.imshow(img[0].permute(1, 2, 0).cpu().detach().numpy()); plt.show()
            if idx == 0: enc2 = e2
            else: enc2 = torch.add(enc2, e2)

            e3 = self.encoder3(self.pool2(e2))
            f2 = self.features[1](f1)
            e3 = self.inter_cons[1](torch.cat((f2, e3), 1))

            if idx == 0: enc3 = e3
            else: enc3 = torch.add(enc3, e3)

            e4 = self.encoder4(self.pool3(e3))
            del e3
            f3 = self.features[2](f2)
            e4 = self.inter_cons[2](torch.cat((f3, e4), 1))

            if idx == 0: enc4 = e4
            else: enc4 = torch.add(enc4, e4)

            bn = self.bottleneck[0](self.pool4(e4))
            del e4
            f4 = self.features[3](f3)
            bn = self.inter_cons[3](torch.cat((f4, bn), 1))

            bn = self.bottleneck[1](bn)
            f5 = self.features[4](f4)
            bn = self.inter_cons[4](torch.cat((f5, bn), 1))

            if idx == 0: bottleneck = bn
            else: bottleneck = torch.add(bottleneck, bn)

            del bn

        del e2
        # Averaging
        enc2 = torch.div(enc2, cfg.general.shots)
        enc3 = torch.div(enc3, cfg.general.shots)
        enc4 = torch.div(enc4, cfg.general.shots)
        bottleneck = torch.div(bottleneck, cfg.general.shots)

        # enc1 = self.encoder1(img)
        # enc2 = self.encoder2(self.pool1(enc1))
        # enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))
        #
        # bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.lcf_conv(dec1))

def _block(in_channels, features, name):
    return nn.Sequential(*[
        nn.Sequential(OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    )
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
            ]
        )),
        nn.Sequential(OrderedDict(
            [
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        ))
    ])