import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import h5py


class OvuleDset(torch_data.Dataset):

    def __init__(self, root, files, shuffle=False):

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])

        self.rawGt = h5py.File(os.path.join(root, files[0]), 'r')
        self.mcSeg = h5py.File(os.path.join(root, files[1]), 'r')

        qlabelTrue = self.rawGt['label'][:].astype(np.float32)
        qlabelFalse = self.mcSeg['exported_data'][:].astype(np.float32)
        qraw = self.rawGt['raw'][:].astype(np.float32)
        labelVals = np.unique(qlabelTrue)
        # plt.imshow(qraw * (qlabel == 0));plt.show()
        maskedTrueCells = []
        midPtsTrueCells = []
        bboxesFalseCells = []
        for labelVal in labelVals[1:len(labelVals)]:
            mask = (qlabelTrue == labelVal)
            maskedRaw = mask * qraw
            ymin, ymax, xmin, xmax, zmin, zmax, midPt = self.bbox(mask)
            bboxesFalseCells.append([ymin, ymax, xmin, xmax, zmin, zmax])
            midPtsTrueCells.append(midPt)
            maskedTrueCells.append(torch.nn.functional.upsample(self.transform(maskedRaw[ymin:ymax + 1, xmin:xmax + 1, zmin:zmax + 1]).unsqueeze(0).unsqueeze(0), size=(100, 100, 100), mode='bicubic'))

        maskedFalseCells = []
        midPtsFalseCells = []
        bboxesFalseCells = []
        labelVals = np.unique(qlabelFalse)
        for labelVal in labelVals[1:len(labelVals)]:
            mask = (qlabelFalse == labelVal)
            maskedRaw = mask * qraw
            ymin, ymax, xmin, xmax, zmin, zmax, midPt = self.bbox(mask)
            bboxesFalseCells.append([ymin, ymax, xmin, xmax, zmin, zmax])
            midPtsFalseCells.append(midPt)
            maskedFalseCells.append(torch.nn.functional.upsample(self.transform(maskedRaw[ymin:ymax + 1, xmin:xmax + 1, zmin:zmax + 1]).unsqueeze(0).unsqueeze(0), size=(100, 100, 100), mode='bicubic'))

        # # this selects only cells as false cells which have a certain similarity to true cells. However this might take quite long
        # falseCellIndices = []
        # for idx1, falseCellMidPt in enumerate(midPtsFalseCells):
        #     for idx2, trueCellMidPt in enumerate(midPtsTrueCells):
        #         if np.linalg.norm(trueCellMidPt-falseCellMidPt) < 8 and np.linalg.norm(midPtsFalseCells[idx1] - midPtsTrueCells[idx1]) < 10:
        #             falseCellIndices.append(idx1)

        dataTrue = torch.nn.upsample(maskedTrueCells, size=(100, 100, 100), mode='bicubic')
        dataFalse = torch.nn.upsample(maskedFalseCells, size=(100, 100, 100), mode='bicubic')
        labelsTrue = torch.ones(len(self.dataTrue))
        labelsFalse = -torch.ones(len(self.dataFalse))
        self.data = torch.cat([dataFalse, dataTrue], 0)
        self.labels = torch.cat([labelsFalse, labelsTrue], 0)
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.labels[idx], self.data[idx]

    def bbox(self, array3d):
        y = np.any(array3d, axis=1)
        x = np.any(array3d, axis=0)
        z = np.any(array3d, axis=0)
        ymin, ymax = np.where(y)[0][[0, -1]]
        xmin, xmax = np.where(x)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        midPt = np.ndarray([(ymax-ymin)//2, (xmax-xmin)//2, (zmax-zmin)//2])  # center pt
        return ymin, ymax, xmin, xmax, zmin, zmax, midPt


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

    def __len__(self):
        return self.length

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