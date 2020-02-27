import numpy as np
import torch
import elf


def calculate_naive_gt_edge_costs(edges, sp_gt):
    return (sp_gt.squeeze()[edges.astype(np.int)][:, 0] != sp_gt.squeeze()[edges.astype(np.int)][:, 1]).float()


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
            n_obj_pnlty = - abs(len(n_obj_new) - len(n_obj_gt)) * 10
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
        y_vals.append([ymin, ymax])
        x_vals.append([xmin, xmax])
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
    return elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()


def check_no_singles(edges, num_nodes):
    return all(np.unique(edges.ravel()) == np.array(range(num_nodes)))


def collate_graphs(node_features, edge_features, edges, shuffle=False):
    for i in len(node_features):
        edges[i] += i
    return torch.stack(node_features), torch.stack(edges), torch.stack(edge_features)
