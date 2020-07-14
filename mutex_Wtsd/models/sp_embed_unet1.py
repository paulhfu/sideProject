from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import _pca_project, _pca_project_1d, plt_bar_plot


class SpVecsUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, device=None, writer=None):
        super(SpVecsUnet, self).__init__()
        self.embed_model = MediumUNet(n_channels, n_classes, device=device)
        self.device = device
        self.writer = writer
        self.writer_counter = 0

    def forward(self, raw):
        import matplotlib.pyplot as plt
        # plt.imshow(raw[0, 0].cpu());plt.show()
        return self.embed_model(raw)

    def get_node_features(self, raw, features, sp_indices, post_input=False):
        sp_feat_vecs = torch.empty((len(sp_indices), features.shape[0])).to(self.device).float()
        for i, sp in enumerate(sp_indices):
            sp = sp.to(self.device)
            mass = len(sp)
            assert mass > 0
            # ival = torch.index_select(features.squeeze(), 1, sp[:, -2].long())
            # sp_features = torch.gather(ival, 2, torch.stack([sp[:, -1].long() for i in range(ival.shape[0])], dim=0).unsqueeze(-1)).squeeze()
            sp_features = features[:, sp[:, -2], sp[:, -1]]
            sp_feat_vecs[i] = sp_features.sum(-1) / mass

        if self.writer is not None and post_input:
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(_pca_project(features.detach().squeeze().cpu().numpy()), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)

            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(raw.cpu().squeeze(), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/state1", fig, self.writer_counter)

            self.writer_counter += 1

        return sp_feat_vecs