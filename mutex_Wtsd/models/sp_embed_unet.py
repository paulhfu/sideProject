from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch


class SpVecsUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, device=None):
        super(SpVecsUnet, self).__init__()
        self.embed_model = MediumUNet(n_channels, n_classes, device=device)
        self.device = device
    # def forward(self, raw, stacked_superpixels=None):
    #     if stacked_superpixels is None:
    #         return super().forward(raw)
    #     features = super().forward(raw)
    #     sp_feat_vecs = torch.empty((len(stacked_superpixels), features.shape[1]))
    #     for i, ssp in enumerate(stacked_superpixels):
    #         ssp = ssp.to(self.device).float()
    #         mass = ssp.sum()
    #         # if mass < ssp.numel():
    #         #     ssp = torch.nn.functional.pad(ssp.unsqueeze(0).unsqueeze(0).unsqueeze(0), (0,0,0,0,0,511), 'replicate').squeeze()
    #         #     ssp = torch.sparse.FloatTensor(ssp.nonzero().transpose(0, 1).to(self.device),
    #         #                                    torch.ones((512 * mass).long().item()).to(self.device),
    #         #                                    torch.Size(ssp.shape))
    #         #     sp_feat_vecs[i] = torch.Tensor.sparse_mask(features.squeeze(), ssp.coalesce()).to_dense().sum((-2, -1))
    #         # else:
    #         ssp = ssp / mass
    #         masked = features.squeeze() * ssp
    #         sp_feat_vecs[i] = masked.sum((-2, -1))
    #     return sp_feat_vecs

    def forward(self, raw, sp_indices=None):
        import matplotlib.pyplot as plt
        # plt.imshow(raw[0, 0].cpu());plt.show()
        features = self.embed_model(raw)
        if sp_indices is None:
            return features
        sp_feat_vecs = torch.empty((len(sp_indices), features.shape[1])).to(self.device).float()
        for i, sp in enumerate(sp_indices):
            mass = len(sp)
            ival = torch.index_select(features.squeeze(), 1, sp[:, 0].long())
            sp_features = torch.gather(ival, 2, torch.stack([sp[:, 1].long() for i in range(ival.shape[0])], dim=0).unsqueeze(-1)).squeeze()
            sp_feat_vecs[i] = sp_features.sum(-1) / mass
        return sp_feat_vecs