from models.simple_unet import UNet, MediumUNet
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import _pca_project, _pca_project_1d, plt_bar_plot
from models.GCNNs.cstm_message_passing import NodeConv1, EdgeConv2


class SpVecsUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, device=None, writer=None):
        super(SpVecsUnet, self).__init__()
        self.embed_model = MediumUNet(n_channels, n_classes, device=device)
        self.device = device
        self.writer = writer
        self.writer_counter = 0
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

    def forward(self, raw, sp_indices=None, post_input=False):
        import matplotlib.pyplot as plt
        # plt.imshow(raw[0, 0].cpu());plt.show()
        features = self.embed_model(raw)
        if sp_indices is None:
            return features
        sp_feat_vecs = torch.empty((len(sp_indices), features.shape[1])).to(self.device).float()
        for i, sp in enumerate(sp_indices):
            mass = len(sp)
            assert mass > 0
            ival = torch.index_select(features.squeeze(), 1, sp[:, 0].long())
            sp_features = torch.gather(ival, 2, torch.stack([sp[:, 1].long() for i in range(ival.shape[0])], dim=0).unsqueeze(-1)).squeeze()
            sp_feat_vecs[i] = sp_features.sum(-1) / mass

        if self.writer is not None and post_input:
            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(_pca_project(features.detach().squeeze().cpu().numpy()), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)

            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(raw.unsqueeze(0).cpu().squeeze(), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/state1", fig, self.writer_counter)

            self.writer_counter += 1

        return sp_feat_vecs, features


class SpVecsUnetGcn(nn.Module):
    def __init__(self, n_embedding_channels, n_channels=1, n_classes=10, device=None, writer=None):
        super(SpVecsUnetGcn, self).__init__()
        self.embed_model = MediumUNet(n_channels, n_classes, device=device)
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_embedding_channels = n_embedding_channels
        n_node_feat_channels = self.n_embedding_channels
        self.node_conv1 = NodeConv1(n_node_feat_channels, n_node_feat_channels, n_hidden_layer=5)
        self.edge_conv1 = EdgeConv2(n_node_feat_channels, n_node_feat_channels, 3 * n_node_feat_channels, n_hidden_layer=5)
        self.node_conv2 = NodeConv1(n_node_feat_channels, n_node_feat_channels, n_hidden_layer=5)
        self.edge_conv2 = EdgeConv2(self.n_embedding_channels, n_node_feat_channels, 3 * n_node_feat_channels,
                                    use_init_edge_feats=True, n_init_edge_channels=3 * n_node_feat_channels,
                                    n_hidden_layer=5)

    def forward(self, raw, edge_index, angles, gt_edges, sp_indices=None, post_input=False):
        import matplotlib.pyplot as plt
        # plt.imshow(raw[0, 0].cpu());plt.show()
        embeddings = self.embed_model(raw)
        if sp_indices is None:
            return embeddings
        node_features = torch.empty((len(sp_indices), embeddings.shape[1])).to(self.device).float()
        for i, sp in enumerate(sp_indices):
            mass = len(sp)
            assert mass > 0
            ival = torch.index_select(embeddings.squeeze(), 1, sp[:, 0].long())
            sp_features = torch.gather(ival, 2, torch.stack([sp[:, 1].long() for i in range(ival.shape[0])], dim=0).unsqueeze(-1)).squeeze()
            node_features[i] = sp_features.sum(-1) / mass

        embeddings = embeddings.squeeze()

        node_features, _ = self.node_conv1(node_features, edge_index, angles)
        side_loss1, edge_features = self.edge_conv1(node_features, edge_index)

        node_features, _ = self.node_conv2(node_features, edge_index, angles)
        side_loss2, edge_features = self.edge_conv2(node_features, edge_index, edge_features)

        edge_features = nn.functional.leaky_relu(edge_features)

        side_loss = (side_loss1 + side_loss2) / 2

        if self.writer is not None and post_input:
            # with torch.no_grad():
            #     points = self.mean_shift(embeddings.reshape([-1, embeddings.shape[0]]))
            #     points = points.reshape(embeddings.shape)
            # plt.clf()
            # fig = plt.figure(frameon=False)
            # plt.imshow(_pca_project(points.detach().squeeze().cpu().numpy()), cmap='hot')
            # plt.colorbar()
            # self.writer.add_figure("image/embedding_ms_proj", fig, self.writer_counter)
            pca_proj_edge_fe = _pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PCA1', 'PCA2', 'PCA3'])
            self.writer.add_figure("bar/embedding_proj_edge_features", fig, self.writer_counter)

            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(_pca_project(embeddings.detach().squeeze().cpu().numpy()), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/embedding_proj", fig, self.writer_counter)

            plt.clf()
            fig = plt.figure(frameon=False)
            plt.imshow(raw.unsqueeze(0).cpu().squeeze(), cmap='hot')
            plt.colorbar()
            self.writer.add_figure("image/state1", fig, self.writer_counter)

            self.writer_counter += 1

        return edge_features, embeddings, side_loss
