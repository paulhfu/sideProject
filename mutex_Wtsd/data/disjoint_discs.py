import torch
import matplotlib.pyplot as plt

import torch.utils.data as torch_data
import elf
import numpy as np
from affogato.affinities import compute_affinities
import torchvision.transforms as transforms
from utils.general import calculate_gt_edge_costs
import torch_geometric as tg
import elf.segmentation.features as feats
from affogato.segmentation.mws import get_valid_edges
from mutex_watershed import compute_mws_segmentation_cstm
from matplotlib import cm
import utils.general as gutils
import utils.affinities as affutils
import os
from utils.affinities import get_edge_features_1d, get_stacked_node_data
import h5py
import multiprocessing


class MultiDiscSpGraphDset(tg.data.Dataset):
    def __init__(self, no_suppix=True, length=50000, shape=(128, 128), radius=72, less=False):
        self.mp = (56, 56)
        self.less = less
        self.length = length
        self.shape = shape
        self.no_suppix = no_suppix
        self.radius = radius
        self.fidx = 0
        self.transform = None
        # lock = multiprocessing.Lock()
        self.offsets = [[0, -1], [-1, 0],
                        [-3, 0], [0, -3]]
        self.sep_chnl = 2

        n_disc = 10
        self.rads = []
        self.mps = []
        for disc in range(n_disc):
            radius = np.random.randint(max(self.shape) // 15, max(self.shape) // 10)
            touching = True
            while touching:
                mp = np.array([np.random.randint(0 + radius, self.shape[0] - radius),
                               np.random.randint(0 + radius, self.shape[1] - radius)])
                touching = False
                for other_rad, other_mp in zip(self.rads, self.mps):
                    diff = mp - other_mp
                    if (diff ** 2).sum() ** .5 <= radius + other_rad + 2:
                        touching = True
            self.rads.append(radius)
            self.mps.append(mp)

        return

    def __len__(self):
        return self.length

    def get(self, idx):
        n_disc = np.random.randint(8, 10)
        rads = []
        mps = []
        for disc in range(n_disc):
            radius = np.random.randint(max(self.shape) // 18, max(self.shape) // 15)
            touching = True
            while touching:
                mp = np.array([np.random.randint(0 + radius, self.shape[0] - radius),
                               np.random.randint(0 + radius, self.shape[1] - radius)])
                touching = False
                for other_rad, other_mp in zip(rads, mps):
                    diff = mp - other_mp
                    if (diff ** 2).sum() ** .5 <= radius + other_rad + 2:
                        touching = True
            rads.append(radius)
            mps.append(mp)

        # take static image
        # rads = self.rads
        # mps = self.mps

        data = np.zeros(shape=self.shape, dtype=np.float)
        gt = np.zeros(shape=self.shape, dtype=np.float)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                bg = True
                for radius, mp in zip(rads, mps):
                    ly, lx = y - mp[0], x - mp[1]
                    if (ly ** 2 + lx ** 2) ** .5 <= radius:
                        data[y, x] += np.cos(np.sqrt((x - self.shape[1]) ** 2 + y ** 2) * 50 * np.pi / self.shape[1])
                        data[y, x] += np.cos(np.sqrt(x ** 2 + y ** 2) * 50 * np.pi / self.shape[1])
                        # data[y, x] += 6
                        gt[y, x] = 1
                        bg = False
                if bg:
                    data[y, x] += np.cos(y * 40 * np.pi / self.shape[0])
                    data[y, x] += np.cos(np.sqrt(x ** 2 + (self.shape[0] - y) ** 2) * 30 * np.pi / self.shape[1])
        data += 1
        # plt.imshow(data);plt.show()
        # if self.no_suppix:
        #     raw = torch.from_numpy(data).float()
        #     return raw.unsqueeze(0), torch.from_numpy(gt.astype(np.long))
            # return torch.stack((torch.rand_like(raw), raw, torch.rand_like(raw))), torch.from_numpy(gt.astype(np.long))

        affinities = affutils.get_naive_affinities(data, self.offsets)
        gt_affinities, _ = compute_affinities(gt == 1, self.offsets)
        gt_affinities[self.sep_chnl:] *= -1
        gt_affinities[self.sep_chnl:] += +1
        affinities[self.sep_chnl:] *= -1
        affinities[self.sep_chnl:] += +1
        # affinities[:self.sep_chnl] /= 1.1
        affinities[self.sep_chnl:] *= 1.01
        affinities = (affinities - (affinities * gt_affinities)) + gt_affinities

        # affinities[self.sep_chnl:] *= -1
        # affinities[self.sep_chnl:] += +1
        # affinities[self.sep_chnl:] *= 4
        affinities = affinities.clip(0, 1)

        valid_edges = get_valid_edges((len(self.offsets),) + self.shape, self.offsets,
                                      self.sep_chnl, None, False)
        node_labeling, neighbors, cutting_edges, mutexes = compute_mws_segmentation_cstm(affinities.ravel(),
                                                                                         valid_edges.ravel(),
                                                                                         self.offsets,
                                                                                         self.sep_chnl,
                                                                                         self.shape)
        node_labeling = node_labeling - 1
        # rag = elf.segmentation.features.compute_rag(np.expand_dims(node_labeling, axis=0))
        # neighbors = rag.uvIds()
        i = 0

        # node_labeling = gt * 5000 + node_labeling
        # segs = np.unique(node_labeling)
        #
        # new_labeling = np.zeros_like(node_labeling)
        # for seg in segs:
        #     i += 1
        #     new_labeling += (node_labeling == seg) * i
        #
        # node_labeling = new_labeling - 1

        # gt_labeling, _, _, _ = compute_mws_segmentation_cstm(gt_affinities.ravel(),
        #                                                      valid_edges.ravel(),
        #                                                      offsets,
        #                                                      self.shape)
        #                                                      self.sep_chnl,

        nodes = np.unique(node_labeling)
        try:
            assert all(nodes == np.array(range(len(nodes)), dtype=np.float))
        except:
            Warning("node ids are off")

        noisy_affinities = np.random.rand(*affinities.shape)
        noisy_affinities = noisy_affinities.clip(0, 1)
        noisy_affinities = affinities

        edge_feat, neighbors = get_edge_features_1d(node_labeling, self.offsets, noisy_affinities)
        gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())

        if self.less:
            raw = torch.from_numpy(data).float()
            node_labeling = torch.from_numpy(node_labeling.astype(np.float32))
            gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.long))
            edges = torch.from_numpy(neighbors.astype(np.long))
            edges = edges.t().contiguous()
            edges = torch.cat((edges, torch.stack((edges[1], edges[0]))), dim=1)
            return raw.unsqueeze(0), node_labeling, torch.from_numpy(gt.astype(np.long)), gt_edge_weights, edges

        # affs = np.expand_dims(affinities, axis=1)
        # boundary_input = np.mean(affs, axis=0)
        # gt1 = gutils.multicut_from_probas(node_labeling.astype(np.float32), neighbors.astype(np.float32),
        #                                  gt_edge_weights.astype(np.float32), boundary_input.astype(np.float32))

        # plt.imshow(node_labeling)
        # plt.show()
        # plt.imshow(gt1)
        # plt.show()

        gt = torch.from_numpy(gt.astype(np.float32)).squeeze().float()

        edges = torch.from_numpy(neighbors.astype(np.long))
        raw = torch.tensor(data).squeeze().float()
        noisy_affinities = torch.tensor(noisy_affinities).squeeze().float()
        edge_feat = torch.from_numpy(edge_feat.astype(np.float32))
        nodes = torch.from_numpy(nodes.astype(np.float32))
        node_labeling = torch.from_numpy(node_labeling.astype(np.float32))
        gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.float32))
        diff_to_gt = (edge_feat[:, 0] - gt_edge_weights).abs().sum().item()
        # node_features, angles = get_stacked_node_data(nodes, edges, node_labeling, raw, size=[32, 32])

        # file = h5py.File("/g/kreshuk/hilt/projects/rags/" + "rag_" + str(self.fidx) + ".h5", "w")
        # file.create_dataset("edges", data=edges.numpy())
        # self.fidx += 1

        if self.no_suppix:
            raw = torch.from_numpy(data).float()
            return raw.unsqueeze(0), torch.from_numpy(gt.numpy().astype(np.long))

        edges = edges.t().contiguous()
        edges = torch.cat((edges, torch.stack((edges[1], edges[0]))), dim=1)

        # print('imbalance: ', abs(gt_edge_weights.sum() - (len(gt_edge_weights) / 2)))

        return edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, noisy_affinities, gt


if __name__ == "__main__":
    set = MultiDiscSpGraphDset(no_suppix=False)
    ret = set.get(3)
    a=1