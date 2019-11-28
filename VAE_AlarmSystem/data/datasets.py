import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import h5py


class OvuleDset(torch_data.Dataset):

    def __init__(self, root, files, mode='train'):
        self.transform = transforms.Compose([
            transforms.Normalize((0,), (1,)),
        ])

        self.trueCells = h5py.File(os.path.join(root, files[0]), 'r')[mode]
        self.badSegCells = h5py.File(os.path.join(root, files[1]), 'r')[mode]

        keysTrueCells = list(self.trueCells.keys())
        keysBadSegCells = list(self.badSegCells.keys())

        for key in keysTrueCells:
            if len(self.trueCells[key][:].shape) != 3:
                del self.trueCells[key]

        for key in keysBadSegCells:
            if len(self.badSegCells[key][:].shape) != 3:
                del self.badSegCells[key]

        labelTrue = torch.zeros(len(keysTrueCells))
        labelFalse = torch.ones(len(keysBadSegCells))

        self.labels = torch.cat([labelFalse, labelTrue], 0)
        self.keys = keysBadSegCells + keysTrueCells

        self.length = len(self.labels)
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx]
        if label == 0:
            if len(torch.tensor(self.trueCells[self.keys[idx]][:]).float().shape) != 3:
                a=1
            return torch.tensor(self.trueCells[self.keys[idx]][:]).float().unsqueeze(0), label.long()
        if label == 1:
            if len(torch.tensor(self.badSegCells[self.keys[idx]][:]).float().shape) != 3:
                a=1
            return torch.tensor(self.badSegCells[self.keys[idx]][:]).float().unsqueeze(0), label.long()


def splitTrainTestSets(root, file):
    dsets = []
    supGroup = h5py.File(os.path.join(root, file), 'a')

    for key in supGroup.keys():
        dsets.append(supGroup[key])
        del supGroup[key]
    testInd = np.random.randint(0, len(dsets), len(dsets)//4)
    train = supGroup.create_group("train")
    test = supGroup.create_group("test")
    for idx, tidx in enumerate(testInd):
        test.create_dataset(f'{idx}', data=dsets[tidx])
    for idx in sorted(testInd, reverse=True):
        del dsets[idx]
    for idx, dset in enumerate(dsets):
        train.create_dataset(f'{idx}', data=dset)
    supGroup.close()
    return

def createOvuleData(root, files, tgtFiles):
    rawGt = h5py.File(os.path.join(root, files[0]), 'r')
    mcSeg = h5py.File(os.path.join(root, files[1]), 'r')

    qlabelTrue = rawGt['label'][:].squeeze().astype(np.uint16, casting='safe')
    qlabelFalse = mcSeg['exported_data'][:].squeeze()
    qraw = rawGt['raw'][:].squeeze()
    labelVals = np.unique(qlabelTrue)
    # plt.imshow(qraw * (qlabel == 0));plt.show()
    midPtsTrueCells = []
    bboxesTrueCells = []
    labelVals = labelVals[1:-1]
    numTrueCells = len(labelVals)
    chunk = len(labelVals) // 20
    f_trueCells = h5py.File(os.path.join(root, tgtFiles[0]), 'w')
    for i in range(20):
        maskedTrueCells = ()
        for labelVal in tqdm(labelVals[chunk*i:chunk*(i+1)]):
            mask = (qlabelTrue == labelVal)
            augMask = augmentMask(mask)
            zmin, zmax, ymin, ymax, xmin, xmax, midPt = bbox(mask)
            bboxesTrueCells.append([zmin, zmax, ymin, ymax, xmin, xmax])
            midPtsTrueCells.append(midPt)
            maskedRaw = mask * qraw
            maskedTrueCells += (maskedRaw[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1], )
        for l, itm in enumerate(maskedTrueCells):
            if np.any(np.asarray(itm.shape) > 16) and np.linalg.norm(itm.shape) < 500:
                f_trueCells.create_dataset(f"{l+(i*chunk)}", data=itm)

    f_trueCells.close()

    bboxesFalseCells = []
    labelVals = np.unique(qlabelFalse)
    labelVals = labelVals[1:-1]
    labelVals = np.random.choice(labelVals, (numTrueCells))
    chunk = len(labelVals) // 20
    f_falseCells = h5py.File(os.path.join(root, tgtFiles[1]), 'w')
    for i in range(20):
        maskedFalseCells = ()
        midPtsFalseCells = []
        for labelVal in tqdm(labelVals[chunk*i:chunk*(i+1)]):
            mask = (qlabelFalse == labelVal)
            zmin, zmax, ymin, ymax, xmin, xmax, midPt = bbox(mask)
            bboxesFalseCells.append([zmin, zmax, ymin, ymax, xmin, xmax])
            midPtsFalseCells.append(midPt)
            maskedRaw = mask * qraw
            maskedFalseCells += (maskedRaw[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1], )

        # this selects only cells as false cells which have a certain similarity to true cells. However this might take quite long
        ind_tooGood = []
        for idx1, falseCellMidPt in enumerate(midPtsFalseCells):
            for idx2, trueCellMidPt in enumerate(midPtsTrueCells):
                if np.linalg.norm(trueCellMidPt-falseCellMidPt) < 8:
                    if np.linalg.norm(np.asarray(bboxesFalseCells[idx1]) - np.asarray(bboxesTrueCells[idx2])) < 10:
                        ind_tooGood.append(idx1)

        ind_maskedFalseCells = list(range(0, len(maskedFalseCells)))
        for idx in ind_tooGood:
            try:
                ind_maskedFalseCells.remove(idx)
            except:
                pass

        for l, idx in enumerate(ind_maskedFalseCells):
            if np.any(np.asarray(maskedFalseCells[idx].shape) > 16) and np.linalg.norm(maskedFalseCells[idx].shape) < 500:
                f_falseCells.create_dataset(f"{l+(i*chunk)}", data=maskedFalseCells[idx])

    f_falseCells.close()
    splitTrainTestSets(root, tgtFiles[0])
    splitTrainTestSets(root, tgtFiles[1])

def augmentMask(mask):


    return None

def bbox(array3d):
    z = np.any(array3d, axis=1)
    x = np.any(array3d, axis=0)
    ymin, ymax = np.where(x)[0][[0, -1]]
    xmin, xmax = np.where(x.transpose())[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    midPt = np.asarray([(zmax-zmin)//2, (ymax-ymin)//2, (xmax-xmin)//2])  # center pt
    return zmin, zmax, ymin, ymax, xmin, xmax, midPt


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