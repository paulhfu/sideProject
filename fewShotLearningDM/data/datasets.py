import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import h5py


class OvuleDset(torch_data.Dataset):

    def __init__(self, root, files, x_transform=None, y_transform=None, shuffle=False):
        self.x_transform = x_transform
        self.y_transform = y_transform

        self.group = h5py.File(os.path.join(root, files[0]), 'r')
        all_labels = self.group['label']
        all_raw = self.group['raw']
        self.length = len(self.group['label'])
        # for idx in range(1, len(files)):
        #     self.group = h5py.File(os.path.join(root, files[idx]), 'r')
        #     all_labels = np.concatenate((all_labels, self.group['label']), axis=0)
        #     all_raw = np.concatenate((all_raw, self.group['raw']), axis=0)
        #     self.length += len(self.group['label'])
        all_labels.astype(np.float32)
        all_raw.astype(np.float32)

    def __getitem__(self, idx):
        qlabel = self.group['label'][idx, :, :].astype(np.float32)
        qraw = self.group['raw'][idx, :, :].astype(np.float32)
        labelVals = np.unique(qlabel)
        maskedCells = ()
        for labelVal in labelVals[1:len(labelVals)]:
            mask = (qlabel == labelVal)
            maskedRaw = mask * qraw
            rmin, rmax, cmin, cmax = self.bbox(mask)
            maskedCells += (maskedRaw[rmin:rmax + 1, cmin:cmax + 1],)

        plt.imshow(qlabel);plt.show()
        plt.imshow(qraw);plt.show()
        for cell in maskedCells:
            plt.imshow(cell);plt.show()
        return maskedCells

    def bbox(self, array2d):
        rows = np.any(array2d, axis=1)
        cols = np.any(array2d, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

class TomatoeDset(torch_data.Dataset):

    def __init__(self, root, files, x_transform=None, y_transform=None, shuffle=False):
        self.x_transform = x_transform
        self.y_transform = y_transform

        self.group = h5py.File(os.path.join(root, files[0]), 'r')
        # all_data = self.group['data']
        all_raw = self.group['raw']
        self.length = len(self.group['data'])
        # all_data.astype(np.float32)
        all_raw.astype(np.float32)

    def __getitem__(self, idx):
        # data = self.group['data'][0, idx, :, :, 0].astype(np.float32)
        raw = self.group['raw'][idx, :, :].astype(np.float32)
        return torch.tensor(raw) #, torch.tensor(data)


class MeristemDset(torch_data.Dataset):

    def __init__(self, root, files, x_transform=None, y_transform=None, shuffle=False):
        self.x_transform = x_transform
        self.y_transform = y_transform

        self.group = h5py.File(os.path.join(root, files[0]), 'r')
        # all_data = self.group['data']
        all_raw = self.group['raw']
        self.length = len(self.group['data'])
        # all_data.astype(np.float32)
        all_raw.astype(np.float32)

    def __getitem__(self, idx):
        # data = self.group['data'][0, idx, :, :, 0].astype(np.float32)
        raw = self.group['raw'][idx, :, :].astype(np.float32)
        return torch.tensor(raw) #, torch.tensor(data)