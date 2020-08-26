from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import _pca_project, _pca_project_1d, plt_bar_plot
from models.unet3d.model import UNet2D
from affogato.segmentation import compute_mws_segmentation
from mu_net.criteria.contrastive_loss import ContrastiveLoss


class SpVecsUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, device=None, writer=None):
        super(SpVecsUnet, self).__init__()
        self.embed_model = UNet2D(n_channels, n_classes, final_sigmoid=False, f_maps=n_classes)
        self.device = device
        self.writer = writer
        self.writer_counter = 0
        self.delta_var = 0.5
        self.delta_dist = 1.5
        self.pw_dist = torch.nn.PairwiseDistance()
        self.loss = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)

    def forward(self, raw):
        import matplotlib.pyplot as plt
        # plt.imshow(raw[0, 0].cpu());plt.show()
        raw = raw.unsqueeze(2)
        ret = self.embed_model(raw).squeeze(2)
        return ret

    def get_edges_from_embeddings(self, embeddings, offsets, sep_chnl):
        affinities = []
        for off in offsets:
            dist = torch.norm(embeddings - embeddings[::off[0], ::off[1]])
            affinities.append[torch.clamp((2*self.delta_dist - dist) / 2*self.delta_dist, min=0)]

        affinities[sep_chnl:] = 1 - affinities[sep_chnl:]
        return affinities

    def get_node_features(self, embeddings, _, post_input=False):
        separating_channel = 2
        offsets = [[-1, 0], [0, -1],
           # direct 3d nhood for attractive edges
           [-1, -1], [1, 1], [-1, 1],[1, -1]
           # indirect 3d nhood for dam edges
           [-9, 0], [0, -9],
           # long range direct hood
           [-9, -9], [9, -9], [-9, -4], [-4, -9], [4, -9], [9, -4],
           # inplane diagonal dam edges
           [-27, 0], [0, -27]]

        edges = self.get_edges_from_embeddings(embeddings, offsets, separating_channel)

        node_labeling = compute_mws_segmentation(edges, offsets, separating_channel, strides=[10, 10], randomize_strides=True)

        stacked_superpixels = [node_labeling == n for n in range(node_labeling.max() + 1)]
        sp_indices = [sp.nonzero().cpu() for sp in stacked_superpixels]

        sp_feat_vecs = torch.empty((len(sp_indices), embeddings.shape[0])).to(self.device).float()
        sp_similarity_reg = 0
        for i, sp in enumerate(sp_indices):
            sp = sp.to(self.device)
            mass = len(sp)
            assert mass > 0
            # ival = torch.index_select(features.squeeze(), 1, sp[:, -2].long())
            # sp_features = torch.gather(ival, 2, torch.stack([sp[:, -1].long() for i in range(ival.shape[0])], dim=0).unsqueeze(-1)).squeeze()
            sp_features = embeddings[:, sp[:, -2], sp[:, -1]].T
            # if sp_features.shape[0] > 1:
            #     shift = torch.randint(1, sp_features.shape[0], (1,)).item()
            #     sp_similarity_reg = sp_similarity_reg + self.pw_dist(sp_features, sp_features.roll(shift, dims=0)).sum()/mass
            sp_feat_vecs[i] = sp_features.sum(0) / mass

        if self.writer is not None and post_input:
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(_pca_project(embeddings.detach().squeeze().cpu().numpy()))
            plt.colorbar()
            self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)
            self.writer_counter += 1

        return sp_feat_vecs, sp_similarity_reg