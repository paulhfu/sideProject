import torch.utils.data as torch_data
import torch
import numpy as np
from affogato.affinities import compute_affinities
from skimage import draw
from skimage.filters import gaussian
import torchvision.transforms as transforms
from utils.general import calculate_gt_edge_costs
import torch_geometric as tg
import elf.segmentation.features as feats
from affogato.segmentation.mws import get_valid_edges
from mutex_watershed import compute_mws_segmentation_cstm
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import nifty.graph
import skimage.data
import skimage.segmentation
import vigra
import pylab
import h5py

offsets = [[0, -1], [-1, 0],
           # direct 3d nhood for attractive edges
           # [0, -1], [-1, 0]]
            [-3, 0], [0, -3],
           [-6, 0], [0, -6]]
sep_chnl = 2
np.random.seed(100)

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

class Polys_and_ellis():

    def __init__(self):
        self.dim = (256, 256)
        self.edge_offsets = offsets
        self.sep_chnl = 2
        self.transform = None
        self.n_ellips = 5
        self.n_polys = 10
        self.n_rect = 5
        self.ellips_color = np.array([1, 0, 0], dtype=np.float)
        self.rect_color = np.array([0, 0, 1], dtype=np.float)
        self.col_diff = 0.4
        self.min_r, self.max_r = 10, 20
        self.min_dist = self.max_r
        return

    def __len__(self):
        return 100000

    def get(self, idx):
        img = np.random.randn(*(self.dim + (3,)))/5

        ri1, ri2, ri3, ri4, ri5, ri6 = np.sign(np.random.randint(-100, 100)) * ((np.random.rand() * 2) + .5), np.sign(
            np.random.randint(-100, 100)) * ((np.random.rand() * 2) + .5), (np.random.rand() * 4) + 3, (
                                               np.random.rand() * 4) + 3, np.sign(np.random.randint(-100, 100)) * (
                                                   (np.random.rand() * 2) + .5), np.sign(np.random.randint(-100, 100)) * ((np.random.rand() * 2) + .5)
        x = np.zeros(self.dim)
        x[:, :] = np.arange(img.shape[0])[np.newaxis, :]
        y = x.transpose()
        img += (np.sin(np.sqrt((x*ri1) ** 2 + ((self.dim[1] - y)*ri2) ** 2) * ri3 * np.pi / self.dim[0]))[..., np.newaxis]
        img += (np.sin(np.sqrt((x*ri5) ** 2 + ((self.dim[1] - y)*ri6) ** 2) * ri4 * np.pi / self.dim[1]))[..., np.newaxis]
        img = gaussian(np.clip(img, 0.1, 1), sigma=.8)
        circles = []
        cmps = []
        while len(circles) < self.n_ellips:
            mp = np.random.randint(self.min_r, self.dim[0]-self.min_r, 2)
            too_close = False
            for cmp in cmps:
                if np.linalg.norm(cmp-mp) < self.min_dist:
                    too_close = True
            if too_close:
                continue
            r = np.random.randint(self.min_r, self.max_r, 2)
            circles.append(draw.circle(mp[0], mp[1], r[0], shape=self.dim))
            cmps.append(mp)

        polys = []
        while len(polys) < self.n_polys:
            mp = np.random.randint(self.min_r, self.dim[0]-self.min_r, 2)
            too_close = False
            for cmp in cmps:
                if np.linalg.norm(cmp-mp) < self.min_dist//2:
                    too_close = True
            if too_close:
                continue
            circle = draw.circle_perimeter(mp[0], mp[1], self.max_r)
            poly_vert = np.random.choice(len(circle[0]), np.random.randint(3, 6), replace=False)
            polys.append(draw.polygon(circle[0][poly_vert], circle[1][poly_vert], shape=self.dim))
            cmps.append(mp)

        rects = []
        while len(rects) < self.n_rect:
            mp = np.random.randint(self.min_r, self.dim[0]-self.min_r, 2)
            _len = np.random.randint(self.min_r//2, self.max_r, (2,))
            too_close = False
            for cmp in cmps:
                if np.linalg.norm(cmp - mp) < self.min_dist:
                    too_close = True
            if too_close:
                continue
            start = (mp[0] - _len[0], mp[1]-_len[1])
            rects.append(draw.rectangle(start, extent=(_len[0]*2, _len[1]*2), shape=self.dim))
            cmps.append(mp)

        for poly in polys:
            color = np.random.rand(3)
            while np.linalg.norm(color-self.ellips_color) < self.col_diff or np.linalg.norm(color-self.rect_color) < self.col_diff:
                color = np.random.rand(3)
            img[poly[0], poly[1], :] = color
            img[poly[0], poly[1], :] += np.random.randn(len(poly[1]), 3)/5

        cols = np.random.choice(np.arange(4, 11, 1).astype(np.float)/10, self.n_ellips, replace=False)
        for i, ellipse in enumerate(circles):
            ri1, ri2, ri3, ri4, ri5, ri6 = np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7), np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7), (np.random.rand() + 1) * 3, (
                        np.random.rand() + 1) * 3, np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7), np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7)
            img[ellipse[0], ellipse[1], :] = np.array([cols[i], 0.0, 0.0])
            img[ellipse[0], ellipse[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt((x[ellipse[0], ellipse[1]]*ri5) ** 2 + ((self.dim[1] - y[ellipse[0], ellipse[1]]) * ri2) ** 2) * ri3 * np.pi / self.dim[0]))[..., np.newaxis]*0.15)+0.2
            img[ellipse[0], ellipse[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt((x[ellipse[0], ellipse[1]]*ri6) ** 2 + ((self.dim[1] - y[ellipse[0], ellipse[1]])*ri1) ** 2) * ri4 * np.pi / self.dim[1]))[..., np.newaxis]*0.15)+0.2
            # img[ellipse[0], ellipse[1], :] += np.random.randn(len(ellipse[1]), 3) / 10

        cols = np.random.choice(np.arange(4, 11, 1).astype(np.float)/10, self.n_rect, replace=False)
        for rect in rects:
            ri1, ri2, ri3, ri4, ri5, ri6 = np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7), np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7), (np.random.rand() + 1) * 3, (
                        np.random.rand() + 1) * 3, np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7), np.sign(np.random.randint(-100,100))*((np.random.rand()*4)+7)
            img[rect[0], rect[1], :] = np.array([0.0, 0.0, cols[i]])
            img[rect[0], rect[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt((x[rect[0], rect[1]] * ri5) ** 2 + ((self.dim[1] - y[rect[0], rect[1]]) * ri2) ** 2) * ri3 * np.pi/self.dim[0]))[..., np.newaxis] * 0.15) + 0.2
            img[rect[0], rect[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt((x[rect[0], rect[1]] * ri1) ** 2 + ((self.dim[1] - y[rect[0], rect[1]]) * ri6) ** 2) * ri4 * np.pi/self.dim[1]))[..., np.newaxis] * 0.15) + 0.2
            # img[rect[0], rect[1], :] += np.random.randn(*(rect[1].shape + (3,)))/10

        img = np.clip(img, 0, 1).astype(np.float32)

        smooth_image = gaussian(img, sigma=.2)

        # shape = np.array(smooth_image.shape[0:2]).astype(np.uint32).tolist()
        # taggedImg = vigra.taggedView(smooth_image, 'xyc')
        # edgeStrength = vigra.filters.structureTensorEigenvalues(taggedImg, 1.5, 1.9)[:, :, 0]
        # edgeStrength = edgeStrength.squeeze()
        # edgeStrength = np.array(edgeStrength).astype(np.float32)
        # seeds = vigra.analysis.localMinima(edgeStrength)
        # seeds = vigra.analysis.labelImageWithBackground(seeds)
        # gridGraph = nifty.graph.undirectedGridGraph(shape)
        # # oversegNodeWeighted = nifty.graph.nodeWeightedWatershedsSegmentation(graph=gridGraph, seeds=seeds.ravel(),
        # #                                                                      nodeWeights=edgeStrength.ravel())
        # # oversegNodeWeighted = oversegNodeWeighted.reshape(shape)
        #
        # gridGraphEdgeStrength = gridGraph.imageToEdgeMap(edgeStrength, mode='sum')
        # np.random.permutation(gridGraphEdgeStrength)
        # oversegEdgeWeightedA = nifty.graph.edgeWeightedWatershedsSegmentation(graph=gridGraph, seeds=seeds.ravel(),
        #                                                                    edgeWeights=gridGraphEdgeStrength)
        # oversegEdgeWeightedA = oversegEdgeWeightedA.reshape(shape)
        # interpixelShape = [2 * s - 1 for s in shape]
        # imgBig = vigra.sampling.resize(taggedImg, interpixelShape)
        # edgeStrength = vigra.filters.structureTensorEigenvalues(imgBig, 2 * 1.5, 2 * 1.9)[:, :, 0]
        # edgeStrength = edgeStrength.squeeze()
        # edgeStrength = np.array(edgeStrength)
        # gridGraphEdgeStrength = gridGraph.imageToEdgeMap(edgeStrength, mode='interpixel')
        # oversegEdgeWeightedB = nifty.graph.edgeWeightedWatershedsSegmentation(
        #     graph=gridGraph,
        #     seeds=seeds.ravel(),
        #     edgeWeights=gridGraphEdgeStrength)
        # oversegEdgeWeightedB = oversegEdgeWeightedB.reshape(shape)

        affinities = get_naive_affinities(smooth_image, offsets)
        affinities[:self.sep_chnl] *= -1
        affinities[:self.sep_chnl] += +1
        affinities[:self.sep_chnl] /= 1.3
        affinities[self.sep_chnl:] *= 1.3
        affinities = np.clip(affinities, 0, 1)
        #
        valid_edges = get_valid_edges((len(self.edge_offsets),) + self.dim, self.edge_offsets,
                                           self.sep_chnl, None, False)
        node_labeling, neighbors, cutting_edges, mutexes = compute_mws_segmentation_cstm(affinities.ravel(),
                                                                                    valid_edges.ravel(),
                                                                                    offsets,
                                                                                    self.sep_chnl,
                                                                                    self.dim)

        return img, None




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
        radius = np.random.randint(max(self.shape)//5, max(self.shape)//3)
        mp = (np.random.randint(0+radius, self.shape[0]-radius),
              np.random.randint(0+radius, self.shape[1]-radius))
        # mp = self.mp
        data = np.zeros(shape=self.shape, dtype=np.float)
        gt = np.zeros(shape=self.shape, dtype=np.float)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                ly, lx = y-mp[0], x-mp[1]
                if (ly**2 + lx**2)**.5 <= radius:
                    data[y, x] += np.sin(x * 10 * np.pi / self.shape[1])
                    data[y, x] += np.sin(np.sqrt(x**2 + y**2) * 20 * np.pi / self.shape[1])
                    gt[y, x] = 1
                else:
                    data[y, x] += np.sin(y * 5 * np.pi / self.shape[1])
                    data[y, x] += np.sin(np.sqrt(x ** 2 + (self.shape[1]-y) ** 2) * 10 * np.pi / self.shape[1])
        # plt.imshow(data);plt.show()
        gt_affinities, _ = compute_affinities(gt == 1, offsets)

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
        gt = np.zeros(shape=self.shape, dtype=np.float)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                ly, lx = y-mp[0], x-mp[1]
                if (ly**2 + lx**2)**.5 <= radius:
                    data[y, x] += np.sin(x * 10 * np.pi / self.shape[1])
                    data[y, x] += np.sin(np.sqrt(x**2 + y**2) * 20 * np.pi / self.shape[1])
                    # data[y, x] += 4
                    gt[y, x] = 1
                else:
                    data[y, x] += np.sin(y * 10 * np.pi / self.shape[1])
                    data[y, x] += np.sin(np.sqrt(x ** 2 + (self.shape[1]-y) ** 2) * 10 * np.pi / self.shape[1])
        data += 1
        # plt.imshow(data);plt.show()
        gt_affinities, _ = compute_affinities(gt == 1, offsets)

        seg_arbitrary = np.zeros_like(data)
        square_dict = {}
        i = 0
        granularity = 30
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if (x // granularity, y // granularity) not in square_dict:
                    square_dict[(x // granularity, y // granularity)] = i
                    i += 1
                seg_arbitrary[y, x] += square_dict[(x // granularity, y // granularity)]
        seg_arbitrary += gt * 1000
        i=0
        segs = np.unique(seg_arbitrary)
        seg_arb = np.zeros_like(seg_arbitrary)
        for seg in segs:
            seg_arb += (seg_arbitrary == seg) * i
            i += 1
        seg_arbitrary = seg_arb
        rag = feats.compute_rag(np.expand_dims(seg_arbitrary, axis=0))
        neighbors = rag.uvIds()

        affinities = get_naive_affinities(data, offsets)
        # edge_feat = get_edge_features_1d(seg_arbitrary, offsets, affinities)
        # self.edge_offsets = [[1, 0], [0, 1], [1, 0], [0, 1]]
        # self.sep_chnl = 2
        # affinities = np.stack((ndimage.sobel(data, axis=0), ndimage.sobel(data, axis=1)))
        # affinities = np.concatenate((affinities, affinities), axis=0)
        affinities[:self.sep_chnl] *= -1
        affinities[:self.sep_chnl] += +1
        affinities[self.sep_chnl:] /= 0.2
        #
        raw = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
        # if self.aff_pred is not None:
        #     gt_affinities[self.sep_chnl:] *= -1
        #     gt_affinities[self.sep_chnl:] += +1
        #     gt_affinities[:self.sep_chnl] /= 1.5
            # with torch.set_grad_enabled(False):
            #     affinities = self.aff_pred(raw.to(self.aff_pred.device))
            #     affinities = affinities.squeeze().detach().cpu().numpy()
            #     affinities[self.sep_chnl:] *= -1
            #     affinities[self.sep_chnl:] += +1
            #     affinities[:self.sep_chnl] /= 1.2

        valid_edges = get_valid_edges((len(self.edge_offsets),) + self.shape, self.edge_offsets,
                                           self.sep_chnl, None, False)
        node_labeling, neighbors, cutting_edges, mutexes = compute_mws_segmentation_cstm(affinities.ravel(),
                                                                                    valid_edges.ravel(),
                                                                                    offsets,
                                                                                    self.sep_chnl,
                                                                                    self.shape)
        node_labeling = node_labeling -1
        node_labeling = seg_arbitrary
        # plt.imshow(cm.prism(node_labeling/node_labeling.max()));plt.show()
        # plt.imshow(data);plt.show()
        neighbors = (node_labeling.ravel())[neighbors]
        nodes = np.unique(node_labeling)
        edge_feat = get_edge_features_1d(node_labeling, offsets, affinities)

        # for i, node in enumerate(nodes):
        #     seg = node_labeling == node
        #     masked_data = seg * data
        #     idxs = np.where(seg)
        #     dxs1 = np.stack(idxs).transpose()
        #     # y, x = bbox(np.expand_dims(seg, 0))
        #     # y, x = y[0], x[0]
        #     mass = np.sum(seg)
        #     # _, s, _ = np.linalg.svd(StandardScaler().fit_transform(seg))
        #     mean = np.sum(masked_data) / mass
        #     cm = np.sum(dxs1, axis=0) / mass
        #     var = np.var(data[idxs[0], idxs[1]])
        #
        #     mean = 0 if mean < .5 else 1
        #
        #     node_features[node] = torch.tensor([mean])

        offsets_3d = [[0, 0, -1], [0, -1, 0], [0, -3, 0], [0, 0, -3]]

        # rag = feats.compute_rag(np.expand_dims(node_labeling, axis=0))
        # edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
        # gt_edge_weights = feats.compute_affinity_features(rag, np.expand_dims(gt_affinities, axis=1), offsets_3d)[:, 0]
        gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())
        # gt_edge_weights = utils.calculate_naive_gt_edge_costs(neighbors, node_features).unsqueeze(-1)
        # affs = np.expand_dims(affinities, axis=1)
        # boundary_input = np.mean(affs, axis=0)
        # plt.imshow(multicut_from_probas(node_labeling, neighbors, gt_edge_weights, boundary_input));plt.show()

        # neighs = np.empty((10, 2))
        # gt_neighs = np.empty(10)
        # neighs[0] = neighbors[30]
        # gt_neighs[0] = gt_edge_weights[30]
        # i = 0
        # while True:
        #     for idx, n in enumerate(neighbors):
        #         if n[0] in neighs.ravel() or n[1] in neighs.ravel():
        #             neighs[i] = n
        #             gt_neighs[i] = gt_edge_weights[idx]
        #             i += 1
        #             if i == 10:
        #                 break
        #     if i == 10:
        #         break
        #
        # nodes = nodes[np.unique(neighs.ravel())]
        # node_features = nodes
        # neighbors = neighs

        edges = torch.from_numpy(neighbors.astype(np.long))
        raw = raw.squeeze()
        edge_feat = torch.from_numpy(edge_feat.astype(np.float32))
        nodes = torch.from_numpy(nodes.astype(np.float32))
        # gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.float32))
        # affinities = torch.from_numpy(affinities.astype(np.float32))
        affinities = torch.from_numpy(gt_affinities.astype(np.float32))
        gt_affinities = torch.from_numpy(gt_affinities.astype(np.float32))
        node_labeling = torch.from_numpy(node_labeling.astype(np.float32))

        gt_edge_weights = torch.from_numpy(gt_edge_weights.astype(np.float32))
        # noise = torch.randn_like(edge_feat) / 3
        # edge_feat += noise
        # edge_feat = torch.min(edge_feat, torch.ones_like(edge_feat))
        # edge_feat = torch.max(edge_feat, torch.zeros_like(edge_feat))
        diff_to_gt = (edge_feat[:, 0] - gt_edge_weights).abs().sum()

        node_features, angles = get_stacked_node_data(nodes, edges, node_labeling, raw, size=[32, 32])
        # plt.imshow(node_features.view(-1, 32));
        # plt.show()

        edges = edges.t().contiguous()
        edges = torch.cat((edges, torch.stack((edges[1], edges[0]))), dim=1)

        return edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles


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
        cms[i] = torch.tensor([torch.sum(idxs[0]), torch.sum(idxs[1])]) / mask.sum()
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
            angles[i+len(edges)] = np.pi + angle
        else:
            assert False
    if angles.max() > 2*np.pi+0.00000001 or angles.min()+0.00000001 < 0:
        assert False
    angles = np.rint(angles / (2 * np.pi) * 63)
    return raw_nodes, angles.long()

def get_naive_affinities(raw, offsets):
    affinities = np.zeros([len(offsets)] + list(raw.shape[:2]))
    normed_raw = raw / raw.max()
    for y in range(normed_raw.shape[0]):
        for x in range(normed_raw.shape[1]):
            for i, off in enumerate(offsets):
                if 0 <= y+off[0] < normed_raw.shape[0] and 0 <= x+off[1] < normed_raw.shape[1]:
                    affinities[i, y, x, ...] = np.linalg.norm(normed_raw[y, x] - normed_raw[y+off[0], x+off[1]])
    return affinities

def get_edge_features_1d(sp_seg, offsets, affinities):
    offsets_3d = []
    for off in offsets:
        offsets_3d.append([0] + off)

    rag = feats.compute_rag(np.expand_dims(sp_seg, axis=0))
    edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
    return edge_feat

def interference(ri1, ri2, ri3, ri4):
    dim = (256, 256)
    img = np.random.randn(*(dim + (3,))) / 5
    x = np.zeros(dim)
    x[:, :] = np.arange(img.shape[0])[np.newaxis, :]
    y = x.transpose()
    img += (np.sin(np.sqrt((x*ri1) ** 2 + ((dim[1] - y) * ri2) ** 2) * ri3 * np.pi / dim[0]))[..., np.newaxis]
    # img += (np.sin(np.sqrt((x * ri1) ** 2 + (dim[1] - y) ** 2) * ri4 * np.pi / dim[1]))[..., np.newaxis]

    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    for i in range(3):
        # interference(i, i, 10, 10)
        ds = Polys_and_ellis()
        img, sp = ds.get(1)
        plt.imshow(cm.prism(sp/sp.max()))
        plt.show()
        plt.imshow(img)
        plt.show()
    a=1