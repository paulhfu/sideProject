import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import h5py
import os
from glob import glob
from utils.patch_manager import StridedRollingRandomPatches2D, NoPatches2D

import torch.utils.data as torch_data
import numpy as np

class SpgDset(torch_data.Dataset):
    def __init__(self, root_dir, patch_manager="", patch_stride=None, patch_shape=None):
        self.transform = None
        self.graph_dir = os.path.join(root_dir, 'graph_data')
        self.pix_dir = os.path.join(root_dir, 'pix_data')
        self.graph_file_names = sorted(glob(os.path.join(self.graph_dir, "*.h5")))
        self.pix_file_names = sorted(glob(os.path.join(self.pix_dir, "*.h5")))
        pix_file = h5py.File(self.pix_file_names[0], 'r')
        shape = pix_file["gt"][:].shape
        if patch_manager == "rotated":
            self.pm = StridedRollingRandomPatches2D(patch_stride, patch_shape, shape)
            self.reorder_sp = True
        else:
            self.pm = NoPatches2D()
            self.reorder_sp = False
        self.length = len(self.graph_file_names) * np.prod(self.pm.n_patch_per_dim)
        print('found ', self.length, " data files")

    def __len__(self):
        return self.length

    def viewItem(self, idx):
        pix_file = h5py.File(self.pix_file_names[idx], 'r')
        graph_file = h5py.File(self.graph_file_names[idx], 'r')

        raw = pix_file["raw"][:]
        gt = pix_file["gt"][:]
        sp_seg = graph_file["node_labeling"][:]

        fig, (a1, a2, a3) = plt.subplots(1, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        a1.imshow(raw, cmap='gray')
        a1.set_title('raw')
        a2.imshow(cm.prism(gt/gt.max()))
        a2.set_title('gt')
        a3.imshow(cm.prism(sp_seg/sp_seg.max()))
        a3.set_title('sp')
        plt.tight_layout()
        plt.show()



    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.pm.n_patch_per_dim)
        patch_idx = idx % np.prod(self.pm.n_patch_per_dim)
        pix_file = h5py.File(self.pix_file_names[img_idx], 'r')
        graph_file = h5py.File(self.graph_file_names[img_idx], 'r')

        raw = pix_file["raw"][:]
        if raw.ndim == 2:
            raw = torch.from_numpy(raw).float().unsqueeze(0)
        else:
            raw = torch.from_numpy(raw).permute(2, 0, 1).float()
        gt = torch.from_numpy(pix_file["gt"][:]).unsqueeze(0).float()
        sp_seg = torch.from_numpy(graph_file["node_labeling"][:]).unsqueeze(0).float()

        all = torch.cat([raw, gt, sp_seg], 0)
        patch = self.pm.get_patch(all, patch_idx)

        if not self.reorder_sp:
            return patch[:-2], patch[-2].unsqueeze(0), patch[-1].unsqueeze(0), torch.tensor([img_idx])

        sp_seg = patch[-1].unsqueeze(0)
        new_sp_seg = torch.zeros_like(sp_seg)
        for i, sp in enumerate(torch.unique(sp_seg)):
            new_sp_seg[sp_seg == sp] = i

        return patch[:-2], patch[-2].unsqueeze(0), new_sp_seg, torch.tensor([img_idx])


    def get_graphs(self, indices, device="cpu"):
        edge_weights, edges, edge_feat, diff_to_gt, gt_edge_weights = [], [], [], [], []
        for i in indices:
            graph_file = h5py.File(self.graph_file_names[i], 'r')
            try:
                edges.append(torch.from_numpy(graph_file["edges"][:]).to(device))
                edge_feat.append(torch.from_numpy(graph_file["edge_feat"][:]).to(device))
                diff_to_gt.append(torch.tensor(graph_file["diff_to_gt"][()], device=device))
                gt_edge_weights.append(torch.from_numpy(graph_file["gt_edge_weights"][:]).to(device))
                edge_weights.append(torch.from_numpy(graph_file["affinities"][:]).to(device))
            except:
                a=1

        return edges, edge_feat, diff_to_gt, gt_edge_weights

if __name__ == "__main__":
    set = SpgDset()
    ret = set.get(3)
    a=1