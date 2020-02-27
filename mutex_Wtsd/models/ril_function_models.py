from models.simple_unet import Down, Up, DoubleConv, OutConv
import torch
from torchvision.models.densenet import _Transition, _DenseBlock
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class DNDQN(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, num_inchannels=4, device=None):

        super(DNDQN, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_inchannels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.SmoothL1Loss()
        self.device = device

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class DNDQNRI(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, num_inchannels=4, device=None):

        super(DNDQNRI, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_inchannels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.SmoothL1Loss()
        self.device = device

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return nn.functional.softmax(out, -1)


class UnetFcnDQN(nn.Module):
    def __init__(self, n_inChannels, n_edges, n_actions, bilinear=True, device=None):
        super(UnetFcnDQN, self).__init__()
        self.device = device
        self.bilinear = bilinear
        self.n_edges = n_edges
        self.n_actions = n_actions

        self.inc = DoubleConv(n_inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 128, bilinear)
        self.outc = OutConv(128, n_edges*n_actions)

        self.out_action = OutConv(128, 64)
        self.global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.out_act_lcf = nn.Linear(64, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        ax = self.out_action(x)
        ax = self.global_pool(ax)
        actions = self.out_act_lcf(ax.squeeze(-1).squeeze(-1))
        spatial_sel = self.outc(x)
        shape = spatial_sel.shape
        return spatial_sel.view(shape[0], self.n_edges, self.n_actions, shape[2], shape[3]), actions


class UnetDQN(nn.Module):
    def __init__(self, n_inChannels, n_edges, n_actions, bilinear=True, device=None):
        super(UnetDQN, self).__init__()
        self.device = device
        self.bilinear = bilinear
        self.n_edges = n_edges
        self.n_actions = n_actions

        self.inc = DoubleConv(n_inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 128, bilinear)
        self.outc = OutConv(128, n_edges*n_actions)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        spatial_sel = self.outc(x)
        shape = spatial_sel.shape
        return spatial_sel.view(shape[0], self.n_edges, shape[2], shape[3], self.n_actions)


class UnetRI(nn.Module):
    def __init__(self, n_inChannels, n_edges, n_actions, bilinear=True, device=None):
        super(UnetRI, self).__init__()
        self.device = device
        self.bilinear = bilinear
        self.n_edges = n_edges
        self.n_actions = n_actions

        self.inc = DoubleConv(n_inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 128, bilinear)
        self.outc = OutConv(128, n_edges*n_actions)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        spatial_sel = self.outc(x)
        shape = spatial_sel.shape
        spatial_sel = spatial_sel.view(shape[0], self.n_edges, shape[2], shape[3], self.n_actions)
        return nn.functional.softmax(spatial_sel, -1)


class UnetRI2(nn.Module):
    def __init__(self, n_inChannels, n_edges, n_actions, bilinear=True, device=None):
        super(UnetRI, self).__init__()
        self.device = device
        self.bilinear = bilinear
        self.n_edges = n_edges
        self.n_actions = n_actions

        self.inc = DoubleConv(n_inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 128, bilinear)
        self.outc = OutConv(128, n_edges*n_actions)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        spatial_sel = self.outc(x)
        shape = spatial_sel.shape
        spatial_sel = spatial_sel.view(shape[0], self.n_edges * shape[2], shape[3], self.n_actions)
        return nn.functional.softmax(spatial_sel, -1)


class NodeFeatureExtractor(nn.Module):
    def __init__(self, n_in_channels=2, n_out_channels=16, angular_resolution=64, bilinear=True, device=None):
        super(NodeFeatureExtractor, self).__init__()
        self.device = device
        self.bilinear = bilinear

        self.conv1 = DoubleConv(n_in_channels, 8)
        self.down1 = Down(8, 16)
        self.conv2 = DoubleConv(16, 16)
        self.outc = DoubleConv(16, n_out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.out_lcf = nn.Linear(n_out_channels, angular_resolution)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, x):
        out = self.conv1(x)
        out = self.down1(out)
        out = self.conv2(out)
        spatial_features = self.outc(out)
        ax = self.global_pool(spatial_features)
        angular_weights = self.out_lcf(ax.squeeze(-1).squeeze(-1))
        return spatial_features, angular_weights


class EdgeFeatureExtractor(nn.Module):
    def __init__(self, n_in_channels=2, n_out_channels=16, bilinear=True, device=None):
        super(EdgeFeatureExtractor, self).__init__()
        self.device = device
        self.bilinear = bilinear

        self.conv1 = DoubleConv(n_in_channels, n_out_channels//4)
        self.down1 = Down(n_out_channels//4, n_out_channels//2)
        self.conv2 = DoubleConv(n_out_channels//2, n_out_channels//2)
        self.outc = DoubleConv(n_out_channels//2, n_out_channels)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()

    def forward(self, x):
        out = self.conv1(x)
        out = self.down1(out)
        out = self.conv2(out)
        spatial_features = self.outc(out)
        return spatial_features