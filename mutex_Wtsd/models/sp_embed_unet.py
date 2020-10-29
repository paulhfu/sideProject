from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import _pca_project, _pca_project_1d, plt_bar_plot
from models.unet3d.model import UNet2D

class SpVecsUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, device=None, writer=None):
        super(SpVecsUnet, self).__init__()
        self.embed_model = UNet2D(n_channels, n_classes, final_sigmoid=False, num_levels=5)
        self.device = device
        self.writer = writer
        self.writer_counter = 0
        self.pw_dist = torch.nn.PairwiseDistance()

    def forward(self, raw, post_input=False):
        raw = raw.unsqueeze(2)
        ret = self.embed_model(raw).squeeze(2)
        if self.writer is not None and post_input:
            self.post_pca(ret[0].detach().squeeze())
        return ret

    def get_node_features(self, features, sp_indices):
        sp_feat_vecs = torch.empty((len(sp_indices), features.shape[0])).to(self.device).float()
        for i, sp in enumerate(sp_indices):
            assert len(sp) > 0
            sp_features = features[:, sp[:, -2], sp[:, -1]].T
            sp_feat_vecs[i] = sp_features.mean(0)

        return sp_feat_vecs

    def post_pca(self, features):
        plt.clf()
        fig = plt.figure(frameon=False)
        plt.imshow(_pca_project(features.detach().squeeze().cpu().numpy()))
        self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)
        self.writer_counter += 1
