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
        import matplotlib.pyplot as plt
        raw = raw.unsqueeze(2)
        ret = self.embed_model(raw).squeeze(2)
        if self.writer is not None and post_input:
            self.post_pca(ret[0].detach().squeeze())
        return ret

    def get_node_features(self, features, sp_indices):
        sp_feat_vecs = torch.empty((len(sp_indices), features.shape[0])).to(self.device).float()
        sp_similarity_reg = 0
        for i, sp in enumerate(sp_indices):
            sp = sp.to(self.device)
            mass = len(sp)
            assert mass > 0
            # ival = torch.index_select(features.squeeze(), 1, sp[:, -2].long())
            # sp_features = torch.gather(ival, 2, torch.stack([sp[:, -1].long() for i in range(ival.shape[0])], dim=0).unsqueeze(-1)).squeeze()
            sp_features = features[:, sp[:, -2], sp[:, -1]].T
            # if sp_features.shape[0] > 1:
            #     shift = torch.randint(1, sp_features.shape[0], (1,)).item()
            #     sp_similarity_reg = sp_similarity_reg + self.pw_dist(sp_features, sp_features.roll(shift, dims=0)).sum()/mass
            sp_feat_vecs[i] = sp_features.sum(0) / mass

        return sp_feat_vecs, sp_similarity_reg

    def post_pca(self, features):
        plt.clf()
        fig = plt.figure(frameon=False)
        plt.imshow(_pca_project(features.detach().squeeze().cpu().numpy()))
        self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)
        self.writer_counter += 1
