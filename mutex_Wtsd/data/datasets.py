import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
from affogato.affinities import compute_affinities
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import h5py

offsets = [[0, -1], [-1, 0],
           # direct 3d nhood for attractive edges
           # [0, -1], [-1, 0]]
            [-3, 0], [0, -3]]

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


class CustomDiscDset(torch_data.Dataset):

    def __init__(self, length=5, shape=(56, 56), radius=16):
        self.length = length
        self.shape = shape
        self.radius = radius
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mp = (np.random.randint(0+self.radius, self.shape[0]-self.radius),
              np.random.randint(0+self.radius, self.shape[1]-self.radius))
        data = np.zeros(shape=self.shape, dtype=np.float)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                ly, lx = y-mp[0], x-mp[1]
                if (ly**2 + lx**2)**.5 <= self.radius:
                    data[y, x] = 1
        affinities, _ = compute_affinities(data == 0, offsets)
        return torch.tensor(data).unsqueeze(0).float(), \
               torch.tensor(affinities).float()

