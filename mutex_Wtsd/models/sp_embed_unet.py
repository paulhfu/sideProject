from models.simple_unet import UNet
import torch.nn as nn


class SpVecsUnet(UNet):
    def __init__(self, n_channels=1, n_classes=10, bilinear=True, device=None):
        super(SpVecsUnet, self).__init__(n_channels, n_classes, device)

    def forward(self, raw, stacked_superpixels):
        features = super(raw)
        sp_feat_vecs = []
        for ssp in stacked_superpixels:
            mass = ssp.sum()
            sp_feat_vecs.append((ssp / mass * features).sum())
        return sp_feat_vecs
