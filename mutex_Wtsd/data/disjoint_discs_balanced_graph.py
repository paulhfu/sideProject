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
import h5py
import multiprocessing


class MultiDiscSpGraphDsetBalanced(tg.data.Dataset):
    def __init__(self, no_suppix=True, create=False, shape=(128, 128), length=100000):
        self.length = length
        self.shape = shape
        self.no_suppix = no_suppix
        self.transform = None
        # lock = multiprocessing.Lock()
        self.offsets = [[0, -1], [-1, 0],
                        [-3, 0], [0, -3]]
        self.sep_chnl = 2

        if create:
            self.create_dsets(100)

        return

    def __len__(self):
        return self.length

    def create_dsets(self, num):
        for file_index in range(num):
            n_disc = np.random.randint(25, 30)
            rads = []
            mps = []
            for disc in range(n_disc):
                radius = np.random.randint(max(self.shape) // 25, max(self.shape) // 20)
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
            if self.no_suppix:
                raw = torch.from_numpy(data).float()
                return raw.unsqueeze(0), torch.from_numpy(gt.astype(np.long))
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
            nodes = np.unique(node_labeling)
            try:
                assert all(nodes == np.array(range(len(nodes)), dtype=np.float))
            except:
                Warning("node ids are off")

            noisy_affinities = affinities

            edge_feat, neighbors = get_edge_features_1d(node_labeling, self.offsets, noisy_affinities)
            gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())

            while abs(gt_edge_weights.sum() - (len(gt_edge_weights) / 2)) > 1:
                edge_idx = np.random.choice(np.arange(len(gt_edge_weights)), p=torch.softmax(torch.from_numpy((gt_edge_weights == 0).astype(np.float)), dim=0).numpy())
                if gt_edge_weights[edge_idx] != 0.0:
                    continue

                # print(abs(gt_edge_weights.sum() - (len(gt_edge_weights) / 2)))
                edge = neighbors[edge_idx].astype(np.int)
                # merge superpixel
                diff = edge[0] - edge[1]

                mass = (node_labeling == edge[0]).sum()
                node_labeling = node_labeling - (node_labeling == edge[0]) * diff
                new_mass = (node_labeling == edge[1]).sum()
                try:
                    assert new_mass >= mass
                except:
                    a=1

                # if edge_idx == 0:
                #     neighbors = neighbors[1:]
                #     gt_edge_weights = gt_edge_weights[1:]
                # elif edge_idx == len(gt_edge_weights):
                #     neighbors = neighbors[:-1]
                #     gt_edge_weights = gt_edge_weights[:-1]
                # else:
                #     neighbors = np.concatenate((neighbors[:edge_idx], neighbors[edge_idx+1:]), axis=0)
                #     gt_edge_weights = np.concatenate((gt_edge_weights[:edge_idx], gt_edge_weights[edge_idx+1:]), axis=0)
                #
                # neighbors[neighbors == edge[0]] == edge[1]


                edge_feat, neighbors = get_edge_features_1d(node_labeling, self.offsets, noisy_affinities)
                gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())

            edge_feat, neighbors = get_edge_features_1d(node_labeling, self.offsets, noisy_affinities)
            gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())

            gt = torch.from_numpy(gt.astype(np.float32)).squeeze().float()

            edges = torch.from_numpy(neighbors.astype(np.long))
            raw = torch.tensor(data).squeeze().float()
            noisy_affinities = torch.tensor(noisy_affinities).squeeze().float()
            edge_feat = torch.from_numpy(edge_feat.astype(np.float32))
            nodes = torch.from_numpy(nodes.astype(np.float32))
            node_labeling = torch.from_numpy(node_labeling.astype(np.float32))
            gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.float32))
            diff_to_gt = (edge_feat[:, 0] - gt_edge_weights).abs().sum()
            edges = edges.t().contiguous()
            edges = torch.cat((edges, torch.stack((edges[1], edges[0]))), dim=1)

            self.write_to_h5('/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/data/storage/balanced_graphs/balanced_graph_data' + str(file_index) + '.h5',
                        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, noisy_affinities, gt)


    def get(self, idx):
        idx = np.random.randint(0, 100)
        return self.read_from_h5('/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/data/storage/balanced_graphs/balanced_graph_data' + str(idx) + '.h5')


    def write_to_h5(self, f_name, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, noisy_affinities, gt):
        h5file = h5py.File(f_name, 'w')

        h5file.create_dataset('edges', data=edges.numpy())
        h5file.create_dataset('edge_feat', data=edge_feat.numpy())
        h5file.create_dataset('diff_to_gt', data=diff_to_gt.numpy())
        h5file.create_dataset('gt_edge_weights', data=gt_edge_weights.numpy())
        h5file.create_dataset('node_labeling', data=node_labeling.numpy())
        h5file.create_dataset('raw', data=raw.numpy())
        h5file.create_dataset('nodes', data=nodes.numpy())
        h5file.create_dataset('noisy_affinities', data=noisy_affinities.numpy())
        h5file.create_dataset('gt', data=gt.numpy())

        h5file.close()


    def read_from_h5(self, f_name):
        h5file = h5py.File(f_name, 'r')

        edges = h5file['edges'][:]
        edge_feat = h5file['edge_feat'][:]
        diff_to_gt = h5file['diff_to_gt'][()]
        gt_edge_weights = h5file['gt_edge_weights'][:]
        node_labeling = h5file['node_labeling'][:]
        raw = h5file['raw'][:]
        noisy_affinities = h5file['noisy_affinities'][:]
        gt = h5file['gt'][:]

        h5file.close()

        i = 0
        segs = np.unique(node_labeling)
        new_labeling = np.zeros_like(node_labeling)
        for seg in segs:
            new_labeling += (node_labeling == seg) * i
            i += 1

        node_labeling = new_labeling

        edge_feat, neighbors = get_edge_features_1d(node_labeling, self.offsets, noisy_affinities)
        gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze()).astype(np.float32)

        edges = torch.from_numpy(neighbors.astype(np.long))
        edges = edges.t().contiguous()
        edges = torch.cat((edges, torch.stack((edges[1], edges[0]))), dim=1)

        nodes = np.unique(node_labeling)

        if self.no_suppix:
            raw = torch.from_numpy(raw).float()
            return raw.unsqueeze(0), torch.from_numpy(gt.astype(np.long))
        if self.no_rl:
            raw = torch.from_numpy(raw).float()
            gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.long))
            return raw.unsqueeze(0), torch.from_numpy(gt.astype(np.long)), gt_edge_weights

        print('imbalance: ', abs(gt_edge_weights.sum() - (len(gt_edge_weights) / 2)))

        return edges, torch.from_numpy(edge_feat).float(), diff_to_gt, torch.from_numpy(gt_edge_weights), \
               torch.from_numpy(node_labeling), torch.from_numpy(raw).float(), torch.from_numpy(nodes), \
               torch.from_numpy(noisy_affinities).float(), torch.from_numpy(gt)

def get_stacked_node_data(nodes, edges, segmentation, raw, size):
    raw_nodes = torch.empty([len(nodes), *size])
    cms = torch.empty((len(nodes), 2))
    angles = torch.zeros(len(edges) * 2) - 11
    for i, n in enumerate(nodes):
        mask = (n == segmentation)
        # x, y = utils.bbox(mask.unsqueeze(0).numpy())
        # x, y = x[0], y[0]
        # masked_seg = mask.float() * raw
        # masked_seg = masked_seg[x[0]:x[1]+1, y[0]:y[1]+1]
        # if 0 in masked_seg.shape:
        #     a=1
        # raw_nodes[i] = torch.nn.functional.interpolate(masked_seg.unsqueeze(0).unsqueeze(0), size=size)
        idxs = torch.where(mask)
        cms[n.long()] = torch.tensor([torch.sum(idxs[0]).long(), torch.sum(idxs[1]).long()]) / mask.sum()
    for i, e in enumerate(edges):
        vec = cms[e[1]] - cms[e[0]]
        angle = abs(np.arctan(vec[0] / (vec[1] + np.finfo(float).eps)))
        if vec[0] <= 0 and vec[1] <= 0:
            angles[i] = np.pi + angle
            angles[i + len(edges)] = angle
        elif vec[0] >= 0 and vec[1] <= 0:
            angles[i] = np.pi - angle
            angles[i + len(edges)] = 2 * np.pi - angle
        elif vec[0] <= 0 and vec[1] >= 0:
            angles[i] = 2 * np.pi - angle
            angles[i + len(edges)] = np.pi - angle
        elif vec[0] >= 0 and vec[1] >= 0:
            angles[i] = angle
            angles[i + len(edges)] = np.pi + angle
        else:
            assert False
    if angles.max() > 2 * np.pi + 1e-20 or angles.min() + 1e-20 < 0:
        assert False
    angles = np.rint(angles / (2 * np.pi) * 63)
    return raw_nodes, angles.long()


def get_edge_features_1d(sp_seg, offsets, affinities):
    offsets_3d = []
    for off in offsets:
        offsets_3d.append([0] + off)

    rag = feats.compute_rag(np.expand_dims(sp_seg, axis=0))
    edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
    return edge_feat, rag.uvIds()


if __name__ == "__main__":
    set = MultiDiscSpGraphDset(no_suppix=False)
    ret = set.get(3)
    a=1
