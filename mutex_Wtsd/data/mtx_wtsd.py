import torch
import matplotlib.pyplot as plt

import numpy as np
from affogato.affinities import compute_affinities
from utils.general import calculate_gt_edge_costs
from affogato.segmentation.mws import get_valid_edges
from mutex_watershed import compute_mws_segmentation_cstm
import utils.affinities as affutils
import h5py
from utils.affinities import get_edge_features_1d, get_stacked_node_data


def get_sp_graph(data, gt, scal=1.01):
    offsets = [[0, -1], [-1, 0],
                    [-3, 0], [0, -3]]
    sep_chnl = 2
    shape = (128, 128)

    affinities = affutils.get_naive_affinities(data, offsets)
    gt_affinities, _ = compute_affinities(gt == 1, offsets)
    gt_affinities[sep_chnl:] *= -1
    gt_affinities[sep_chnl:] += +1
    affinities[sep_chnl:] *= -1
    affinities[sep_chnl:] += +1
    affinities[sep_chnl:] *= scal
    affinities = (affinities - (affinities * gt_affinities)) + gt_affinities

    affinities = affinities.clip(0, 1)

    valid_edges = get_valid_edges((len(offsets),) + shape, offsets,
                                  sep_chnl, None, False)
    node_labeling, neighbors, cutting_edges, mutexes = compute_mws_segmentation_cstm(affinities.ravel(),
                                                                                     valid_edges.ravel(),
                                                                                     offsets,
                                                                                     sep_chnl,
                                                                                     shape)
    node_labeling = node_labeling - 1

    nodes = np.unique(node_labeling)
    try:
        assert all(nodes == np.array(range(len(nodes)), dtype=np.float))
    except:
        Warning("node ids are off")

    noisy_affinities = np.random.rand(*affinities.shape)
    noisy_affinities = noisy_affinities.clip(0, 1)
    noisy_affinities = affinities

    edge_feat, neighbors = get_edge_features_1d(node_labeling, offsets, noisy_affinities)
    gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())


    edges = neighbors.astype(np.long)
    noisy_affinities = noisy_affinities.astype(np.float32)
    edge_feat = edge_feat.astype(np.float32)
    nodes = nodes.astype(np.float32)
    node_labeling = node_labeling.astype(np.float32)
    gt_edge_weights = gt_edge_weights.astype(np.float32)
    diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()

    edges = np.sort(edges, axis=-1)
    edges = edges.T
    # edges = np.concatenate((edges, np.stack((edges[1], edges[0]))), axis=1)

    # return node_labeling
    # print('imbalance: ', abs(gt_edge_weights.sum() - (len(gt_edge_weights) / 2)))

    return edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, noisy_affinities