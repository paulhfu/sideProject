import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvNet1d(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, device):
        super(ConvNet1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in_channels, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.lcf = nn.Linear(128, n_out_channels)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()
        self.device = device

    def forward(self, x):
        out = self.conv(x)
        out = self.lcf(out.squeeze())
        return out.squeeze()


class DoubleConv1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.bilinear = bilinear
        # if bilinear, use the normal convolutions to reduce the number of channels
        if not bilinear:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv1d(in_channels, out_channels)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = nn.functional.interpolate(x1, scale_factor=2, mode='linear', align_corners=True)
        else:
            x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet1d(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, bilinear=True, device=None):
        super(UNet1d, self).__init__()
        self.device = device
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1d(n_channels, 64)
        self.down1 = Down1d(64, 128)
        self.down2 = Down1d(128, 256)
        self.down3 = Down1d(256, 512)
        self.down4 = Down1d(512, 512)
        self.up1 = Up1d(1024, 256, bilinear)
        self.up2 = Up1d(512, 128, bilinear)
        self.up3 = Up1d(256, 64, bilinear)
        self.up4 = Up1d(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return nn.functional.relu(logits)