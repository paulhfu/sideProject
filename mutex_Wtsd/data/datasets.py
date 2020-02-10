import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
from affogato.affinities import compute_affinities
import torchvision.transforms as transforms
from mutex_watershed import compute_mws_segmentation_cstm
from sklearn.preprocessing import StandardScaler
from utils import calculate_gt_edge_costs, multicut_from_probas
from affogato.segmentation.mws import get_valid_edges
import torch_geometric as tg
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
from utils import bbox
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import h5py

offsets = [[0, -1], [-1, 0],
           # direct 3d nhood for attractive edges
           # [0, -1], [-1, 0]]
            [-3, 0], [0, -3]]
sep_chnl = 2

def computeAffs(file_from, offsets):
    file = h5py.File(file_from, 'a')
    keys = list(file.keys())
    file.create_group('masks')
    file.create_group('affs')
    for k in keys:
        data = file[k][:].copy()
        affinities, _ = compute_affinities(data != 0, offsets)
        file['affs'].create_dataset(k, data=affinities)
        file['masks'].create_dataset(k, data=data)
        del file[k]
    return

class simpleSeg_4_4_Dset(torch_data.Dataset):

    def __init__(self):
        super(simpleSeg_4_4_Dset, self).__init__()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        simple_img = [[1,1,1,1],[1,0,1,1],[0,0,1,1],[0,0,0,0]]
        simple_affs = [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
                       [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]], [[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]]]
        simple_affs, _ = compute_affinities(np.array(simple_img) == 0, offsets)
        return torch.tensor(simple_img).unsqueeze(0).float(), torch.tensor(simple_affs).unsqueeze(0).float()


class SimpleSeg_20_20_Dset(torch_data.Dataset):

    def __init__(self):
        super(SimpleSeg_20_20_Dset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        img = np.zeros((20, 20))
        affinities = np.ones((len(offsets), 20, 20))
        gt_affinities = np.ones((len(offsets), 20, 20))
        for y in range(len(img)):
            for x in range(len(img[0])):
                if y < 10 and x < 10:
                    img[y,x] = 1
                if y >= 10 and x < 10:
                    img[y, x] = 2
                if y < 10 and x >= 10:
                    img[y, x] = 3
                if y >= 10 and x >= 10:
                    img[y, x] = 4
                if 7 < y < 13 and 7 < x < 13:
                    img[y, x] = 5
        for i in np.unique(img):
            affs, _ = compute_affinities(img == i, offsets)
            gt_affinities *= affs
            # gt_affinities = (gt_affinities == 0).astype(np.float)
        for y in range(len(img)):
            for x in range(len(img[0])):
                if 10 < y < 12 and 10 < x < 12:
                    img[y,x] += 1
                if 16 < y < 18 and 6 < x < 8:
                    img[y, x] += 1
                if 6 < y < 10 and 9 < x < 11:
                    img[y,x] += 1
        for i in np.unique(img):
            affs, _ = compute_affinities(img == i, offsets)
            affinities *= affs
            # affinities = (affinities != 0).astype(np.float)
        affinities = (affinities == 1).astype(np.float)
        gt_affinities = (gt_affinities == 1).astype(np.float)
        affinities[sep_chnl:] *= -1
        affinities[sep_chnl:] += +1
        gt_affinities[sep_chnl:] *= -1
        gt_affinities[sep_chnl:] += +1
        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(affinities).float(), torch.tensor(gt_affinities).float()


class DiscDset(torch_data.Dataset):

    def __init__(self, root, file, mode='train'):
        self.transform = transforms.Compose([
            transforms.Normalize((0,), (1,)),
        ])

        file = h5py.File(os.path.join(root, file), 'r')
        mask_g = file['masks']
        aff_g = file['affs']
        k = list(mask_g.keys())
        self.affs = {}
        self.masks = {}
        for i, key in enumerate(k):
            self.masks[i] = mask_g[key]
            self.affs[i] = aff_g[key]
        self.length = len(k)
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.masks[idx]).unsqueeze(0).float(), \
               torch.tensor(self.affs[idx]).float()

(112,112), 36, (56,56)
(40,40), 10, (20,20)
class CustomDiscDset(torch_data.Dataset):

    def __init__(self, affinities_predictor=None, separating_channel=None, length=50000, shape=(112, 112), radius=36):
        self.mp = (56, 56)
        self.length = length
        self.shape = shape
        self.radius = radius
        self.aff_pred = affinities_predictor
        self.sep_chnl = separating_channel
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mp = (np.random.randint(0+self.radius, self.shape[0]-self.radius),
              np.random.randint(0+self.radius, self.shape[1]-self.radius))
        mp = self.mp
        data = np.zeros(shape=self.shape, dtype=np.float)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                ly, lx = y-mp[0], x-mp[1]
                if (ly**2 + lx**2)**.5 <= self.radius:
                    data[y, x] = 1
        gt_affinities, _ = compute_affinities(data == 0, offsets)

        affinities = gt_affinities
        raw = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
        if self.aff_pred is not None:
            gt_affinities[self.sep_chnl:] *= -1
            gt_affinities[self.sep_chnl:] += +1
            with torch.set_grad_enabled(False):
                affinities = self.aff_pred(raw.to(self.aff_pred.device))
                affinities = affinities.squeeze().detach().cpu().numpy()
                affinities[self.sep_chnl:] *= -1
                affinities[self.sep_chnl:] += +1
                affinities[:self.sep_chnl] /= 1.5

        return raw.squeeze(0), affinities, gt_affinities

class DiscSpGraphDset(tg.data.Dataset):
    def __init__(self, affinities_predictor, separating_channel, edge_offsets,  length=50000, shape=(112, 112), radius=36):
        self.mp = (56, 56)
        self.length = length
        self.shape = shape
        self.edge_offsets = edge_offsets
        self.radius = radius
        self.aff_pred = affinities_predictor
        self.sep_chnl = separating_channel
        self.transform = None
        return

    def __len__(self):
        return 100000

    def get(self, idx):
        radius = np.random.randint(max(self.shape)//5, max(self.shape)//3)
        mp = (np.random.randint(0+radius, self.shape[0]-radius),
              np.random.randint(0+radius, self.shape[1]-radius))
        # mp = self.mp
        data = np.zeros(shape=self.shape, dtype=np.float)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                ly, lx = y-mp[0], x-mp[1]
                if (ly**2 + lx**2)**.5 <= radius:
                    data[y, x] = 1
        gt_affinities, _ = compute_affinities(data == 0, offsets)

        affinities = gt_affinities
        raw = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
        if self.aff_pred is not None:
            gt_affinities[self.sep_chnl:] *= -1
            gt_affinities[self.sep_chnl:] += +1
            gt_affinities[:self.sep_chnl] /= 1.5
            with torch.set_grad_enabled(False):
                affinities = self.aff_pred(raw.to(self.aff_pred.device))
                affinities = affinities.squeeze().detach().cpu().numpy()
                affinities[self.sep_chnl:] *= -1
                affinities[self.sep_chnl:] += +1
                affinities[:self.sep_chnl] /= 1.5

        valid_edges = get_valid_edges((len(self.edge_offsets),) + self.shape, self.edge_offsets,
                                           self.sep_chnl, None, False)
        node_labeling, neighbors, cutting_edges, mutexes = compute_mws_segmentation_cstm(affinities.ravel(),
                                                                                         valid_edges.ravel(),
                                                                                         self.edge_offsets,
                                                                                         self.sep_chnl,
                                                                                         self.shape)
        node_labeling = node_labeling -1
        neighbors = (node_labeling.ravel())[neighbors]
        nodes = np.unique(node_labeling)
        node_features = torch.empty((len(nodes), 3), dtype=torch.float)

        for i, node in enumerate(nodes):
            seg = node_labeling == node
            masked_data = seg * data
            idxs = np.where(seg)
            dxs1 = np.stack(idxs).transpose()
            # y, x = bbox(np.expand_dims(seg, 0))
            # y, x = y[0], x[0]
            mass = np.sum(seg)
            # _, s, _ = np.linalg.svd(StandardScaler().fit_transform(seg))
            mean = np.sum(masked_data) / mass
            cm = np.sum(dxs1, axis=0) / mass
            var = np.var(data[idxs[0], idxs[1]])
            # if mean > 0.0:
            #     print(mean)
            node_features[i] = torch.tensor([mass, mean, var])

        offsets_3d = [[0, 0, -1], [0, -1, 0], [0, -3, 0], [0, 0, -3]]

        rag = feats.compute_rag(np.expand_dims(node_labeling, axis=0))
        edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, 0]
        gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), raw.squeeze().cpu().numpy())
        # affs = np.expand_dims(affinities, axis=1)
        # boundary_input = np.mean(affs, axis=0)
        # plt.imshow(multicut_from_probas(node_labeling, neighbors, gt_edge_weights, boundary_input));plt.show()

        edges = torch.from_numpy(neighbors.astype(np.long))
        edges = edges.t().contiguous()
        edges = torch.cat((edges, torch.stack((edges[1], edges[0]))), dim=1)
        raw = raw.squeeze()
        edge_feat = torch.from_numpy(edge_feat.astype(np.float32)).unsqueeze(-1)
        gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.float32))
        affinities = torch.from_numpy(affinities.astype(np.float32))
        gt_affinities = torch.from_numpy(gt_affinities.astype(np.float32))
        node_labeling = torch.from_numpy(node_labeling.astype(np.float32))
        return edges, edge_feat, gt_edge_weights, node_features, node_labeling, raw.squeeze(), affinities, gt_affinities
