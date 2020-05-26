import numpy as np
import torch
import elf
from torch import multiprocessing as mp
import math
import random
from sklearn.decomposition import PCA


# Global counter
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value


def get_all_arg_combos(grid, paths):
    key = random.choice(list(grid))
    new_paths = []
    new_grid = grid.copy()
    del new_grid[key]
    for val in grid[key]:
        if paths:
            for path in paths:
                path[key] = val
                new_paths.append(path.copy())
        else:
            new_paths.append({key: val})
    if new_grid:
        return get_all_arg_combos(new_grid, new_paths)
    return new_paths



def calculate_naive_gt_edge_costs(edges, sp_gt):
    return (sp_gt.squeeze()[edges.astype(np.int)][:, 0] != sp_gt.squeeze()[edges.astype(np.int)][:, 1]).float()


# Knuth's algorithm for generating Poisson samples
def poisson(self, lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)

# Adjusts learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calculate_gt_edge_costs(neighbors, new_seg, gt_seg):
    rewards = np.zeros(len(neighbors))
    new_seg += 1
    neighbors += 1
    gt_seg += 1

    for idx, neighbor in enumerate(neighbors):
        mask_n1, mask_n2 = new_seg == neighbor[0], new_seg == neighbor[1]
        mask = mask_n1 + mask_n2
        obj_area = np.sum(mask)
        mskd_gt_seg = mask * gt_seg
        mskd_new_seg = mask * new_seg
        n_obj_gt = np.unique(mskd_gt_seg)
        n_obj_new = np.unique(mskd_new_seg)
        n_obj_gt = n_obj_gt[1:] if n_obj_gt[0] == 0 else n_obj_gt
        if len(n_obj_gt) == 1:
            rewards[idx] = 0
        else:
            n_obj_new = n_obj_new[1:] if n_obj_new[0] == 0 else n_obj_new
            assert len(n_obj_new) == 2
            overlaps = np.zeros([len(n_obj_gt)] + [2])
            for j, obj in enumerate(n_obj_gt):
                mask_gt = mskd_gt_seg == obj
                overlaps[j] = np.sum(mask_gt * mask_n1) / np.sum(mask_n1), \
                              np.sum(mask_gt * mask_n2) / np.sum(mask_n2)
            if np.sum(overlaps.max(axis=1) > 0.5) >= 2:
                rewards[idx] = 1
            else:
                rewards[idx] = 0
    new_seg -= 1
    neighbors -= 1
    gt_seg -= 1
    return rewards


def bbox(array2d_c):
    assert len(array2d_c.shape) == 3
    y_vals = []
    x_vals = []
    for array2d in array2d_c:
        y = np.where(np.any(array2d, axis=1))
        x = np.where(np.any(array2d, axis=0))
        ymin, ymax = y[0][[0, -1]] if len(y[0]) != 0 else (0, 0)
        xmin, xmax = x[0][[0, -1]] if len(x[0]) != 0 else (0, 0)
        y_vals.append([ymin, ymax+1])
        x_vals.append([xmin, xmax+1])
    return y_vals, x_vals


def ind_flat_2_spat(flat_indices, shape):
    spat_indices = np.zeros([len(flat_indices)] + [len(shape)], dtype=np.integer)
    for flat_ind, spat_ind in zip(flat_indices, spat_indices):
        rm = flat_ind
        for dim in range(1, len(shape)):
            sz = np.prod(shape[dim:])
            spat_ind[dim - 1] = rm // sz
            rm -= spat_ind[dim - 1] * sz
        spat_ind[-1] = rm
    return spat_indices


def ind_spat_2_flat(spat_indices, shape):
    flat_indices = np.zeros(len(spat_indices), dtype=np.integer)
    for i, spat_ind in enumerate(spat_indices):
        for dim in range(len(shape)):
            flat_indices[i] += max(1, np.prod(shape[dim + 1:])) * spat_ind[dim]
    return flat_indices


def add_rndness_in_dis(dis, factor):
    assert isinstance(dis, np.ndarray)
    assert len(dis.shape) == 2
    ret_dis = dis - ((dis - np.transpose([np.mean(dis, axis=-1)])) * factor)
    return dis


def pca_svd(X, k, center=True):
    # code from https://gist.github.com/project-delphi/e1112dbc0940d729a90f59846d25342b
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n, n])
    H = torch.eye(n) - h
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    explained_variance = torch.mul(s[:k], s[:k])/(n-1)  # remove normalization?
    return components, explained_variance


def multicut_from_probas(segmentation, edges, edge_weights, boundary_input):
    import matplotlib.pyplot as plt
    # for comp in np.unique(segmentation):
    #     plt.imshow(segmentation == comp)
    #     plt.show()
    rag = elf.segmentation.features.compute_rag(np.expand_dims(segmentation, axis=0))
    edge_dict = dict(zip(list(map(tuple, edges)), edge_weights))
    costs = np.empty(len(edge_weights))
    for i, neighbors in enumerate(rag.uvIds()):
        if tuple(neighbors) in edge_dict:
            costs[i] = edge_dict[tuple(neighbors)]
        else:
            costs[i] = edge_dict[(neighbors[1], neighbors[0])]
    edge_sizes = elf.segmentation.features.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = elf.segmentation.multicut.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = elf.segmentation.multicut.multicut_kernighan_lin(rag, costs)
    mc_seg = elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()

    # for comp in np.unique(mc_seg):
    #     plt.imshow(mc_seg == comp)
    #     plt.show()
    return elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()


def check_no_singles(edges, num_nodes):
    return all(np.unique(edges.ravel()) == np.array(range(num_nodes)))


def collate_graphs(node_features, edge_features, edges, shuffle=False):
    for i in len(node_features):
        edges[i] += i
    return torch.stack(node_features), torch.stack(edges), torch.stack(edge_features)


def _pca_project(embeddings):
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=3)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = 3
    img = flattened_embeddings.transpose().reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return np.moveaxis(img.astype('uint8'), 0, -1)
