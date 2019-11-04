import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import h5py


class OvulesDset(torch_data.Dataset):

    def __init__(self, root, files, train=True, x_transform=None, y_transform=None, shuffle=False):
        self.x_transform = x_transform
        self.y_transform = y_transform

        for file in files:
            for idx in range(20,25):
                self.group = h5py.File(os.path.join(root, file), 'r')
                label = self.group['label'][idx, :, :].astype(np.float32)
                raw = self.group['raw'][idx, :, :].astype(np.float32)
                plt.imshow(raw);plt.show()
                plt.imshow(label);plt.show()
        self.length = len(self.group['label'])

    def __getitem__(self, idx):
        label = self.group['label'][idx, :, :].astype(np.float32)
        raw = self.group['raw'][idx, :, :].astype(np.float32)
        return torch.tensor(raw), torch.tensor(label)
