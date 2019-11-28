import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import h5py


class DiscDset(torch_data.Dataset):

    def __init__(self, root, files, mode='train'):
        self.transform = transforms.Compose([
            transforms.Normalize((0,), (1,)),
        ])

        file = h5py.File(os.path.join(root, files[0]), 'r')
        mask = h5py.File(os.path.join(root, files[1]), 'r')
        k = list(file.keys())
        k1 = list(mask.keys())
        self.files = []
        self.masks = []
        if mode == 'train:'
            for i in range(12):
                self.files.append(file[k[i]])
                self.masks.append(mask[k1[i]])
            self.length = 12
        else:
            for i in range(12,len(k)):
                self.files.append(file[k[i]])
                self.masks.append(mask[k1[i]])
            self.length = len(k)-12
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.transform(torch.tensor(self.files[idx])).unsqueeze(0), self.transform(torch.tensor(self.masks[idx])).unsqueeze(0)