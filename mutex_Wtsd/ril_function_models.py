from simple_unet import Down, Up, DoubleConv, OutConv
import torch
import torch.nn as nn

class Q_value(nn.Module):
    def __init__(self, n_inChannels, n_actions, bilinear=True, device=None):
        super(Q_value, self).__init__()
        self.device = device
        self.n_inChannels = n_inChannels
        self.n_actions = n_actions
        self.bilinear = bilinear

        self.inc = DoubleConv(n_inChannels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_actions)

        self.optimizer = torch.optim.SGD(self.parameters(),  lr=0.001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)