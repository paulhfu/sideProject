import torch
import matplotlib.pyplot as plt
import h5py
import os
import portalocker

import torch.utils.data as torch_data
import numpy as np

class SpgDset(torch_data.Dataset):
    def __init__(self, root_dir):
        self.transform = None
        self.graph_dir = os.path.join(root_dir, 'graph_data')
        self.pix_dir = os.path.join(root_dir, 'pix_data')
        self.length = len([name for name in os.listdir(self.graph_dir)])
        print('found ', self.length, " data files")
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pix_file = h5py.File(os.path.join(self.pix_dir, "pix_" + str(idx) + ".h5"), 'r')

        raw = pix_file["raw"][:]
        if raw.ndim == 2:
            raw = torch.from_numpy(raw).float().unsqueeze(0)
        else:
            raw = torch.from_numpy(raw.reshape((raw.shape[-1],) + raw.shape[:-1])).float()
        gt = pix_file["gt"][:]

        return raw, torch.from_numpy(gt).long().unsqueeze(0), torch.tensor([idx])


    def get_graphs(self, indices, device="cpu"):
        edge_weights, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, = [], [], [], [], [], []
        for i in indices:
            graph_file = h5py.File(os.path.join(self.graph_dir, "graph_" + str(i.item()) + ".h5"), 'r')
            try:
                edges.append(torch.from_numpy(graph_file["edges"][:]).to(device))
                edge_feat.append(torch.from_numpy(graph_file["edge_feat"][:]).to(device))
                diff_to_gt.append(torch.tensor(graph_file["diff_to_gt"][()], device=device))
                gt_edge_weights.append(torch.from_numpy(graph_file["gt_edge_weights"][:]).to(device))
                node_labeling.append(torch.from_numpy(graph_file["node_labeling"][:]).to(device))
                edge_weights.append(torch.from_numpy(graph_file["affinities"][:]).to(device))
            except:
                a=1

        node_labeling = torch.stack(node_labeling, 0).unsqueeze(1)
        return edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling

if __name__ == "__main__":
    set = SpgDset()
    ret = set.get(3)
    a=1