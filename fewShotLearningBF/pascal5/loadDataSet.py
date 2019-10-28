from .config import config
import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import os

# in test:
# plane = ['2008_003155.jpg', '2008_001546.jpg', '2008_004704.jpg', '2008_007836.jpg', '2009_000387.jpg', '2009_000440.jpg', '2009_001332.jpg', '2009_001850.jpg', '2009_001851.jpg']
# boat: 2008_005398.jpg  2008_007143.jpg  2009_000469.jpg  2009_000723.jpg  2009_000828.jpg  2009_000919.jpg  2009_002591.jpg  2009_003080.jpg  2009_004125.jpg
# bike: 2008_005727.jpg  2008_008711.jpg  2009_000149.jpg  2009_000573.jpg  2009_001663.jpg  2009_002295.jpg  2009_002928.jpg  2009_003193.jpg  2009_003224.jpg
# bottle: 2009_000121.jpg  2009_000201.jpg  2009_000840.jpg  2009_001818.jpg  2009_002521.jpg  2009_002649.jpg  2009_002749.jpg  2009_002856.jpg  2009_003003.jpg
# bird: 2008_007120.jpg  2008_007994.jpg  2008_008268.jpg  2008_008746.jpg  2009_000037.jpg  2009_000619.jpg  2009_000664.jpg  2009_000879.jpg  2009_000991.jpg

supps = {}
#aeroplane
supps[0] = ['2008_003155', '2008_001546', '2008_004704', '2008_007836', '2009_000387', '2009_000440', '2009_001332', '2009_001850', '2009_001851']
#bycicle
supps[1] = ['2008_005727', '2008_008711', '2009_000149', '2009_000573', '2009_001663', '2009_002295', '2009_002928', '2009_003193', '2009_003224']
#bird
supps[2] = ['2008_007120', '2008_007994', '2008_008268', '2008_008746', '2009_000037', '2009_000619', '2009_000664', '2009_000879', '2009_000991']
#boat
supps[3] = ['2008_005398', '2008_007143', '2009_000469', '2009_000723', '2009_000828', '2009_000919', '2009_002591', '2009_003080', '2009_004125']
#bottle
supps[4] = ['2008_004659', '2009_000201', '2009_000840', '2009_001818', '2009_002521', '2009_002649', '2009_002749', '2009_002856', '2009_003003']
#bus
supps[5] = ['2008_000075', '2010_003293', '2010_005118', '2011_000178', '2011_001064', '2011_000888', '2011_001341', '2008_003141', '2008_002205']
#car
supps[6] = ['2008_002212', '2010_003276', '2010_002929', '2010_003231', '2010_003453', '2010_003813', '2010_004946', '2010_005432', '2010_005860']
#cat
supps[7] = ['2008_001640', '2008_001433', '2008_000464', '2008_000345', '2008_003499', '2008_006325', '2009_004248', '2010_001292', '2010_002900']
#chair
supps[8] = ['2008_002521', '2008_000673', '2008_003821', '2008_004612', '2009_002415', '2009_001299', '2010_003325', '2010_003971', '2011_000813']
#cow
supps[9] = ['2010_004635', '2008_002778', '2008_006063', '2008_005105', '2009_000731', '2008_007031', '2010_002701', '2010_004322', '2011_001232']
#diningtable
supps[10] = ['2008_007048', '2009_002415', '2009_002487', '2009_004140', '2010_001851', '2011_000226', '2010_003971', '2010_003531', '2010_002336']
#dog
supps[11] = ['2008_000863', '2008_001070', '2008_002904', '2008_002936', '2008_003576', '2008_006130', '2008_007513', '2009_000156', '2009_000892']
#horse
supps[12] = ['2010_004519', '2011_001722', '2010_002691', '2009_004942', '2009_003804', '2009_002975', '2009_000705', '2008_008392', '2008_001682']
#motorbike
supps[13] = ['2008_000782', '2009_003196', '2010_001251', '2010_002271', '2010_003207', '2010_004355', '2010_004042', '2010_003947', '2010_005366']
#person
supps[14] = ['2008_000233', '2008_000474', '2008_000213', '2008_001028', '2008_000992', '2008_001074', '2008_001076', '2008_001231', '2008_001170']
#pottedplant
supps[15] = ['', '', '', '', '', '', '', '', '']
#sheep
supps[16] = ['', '', '', '', '', '', '', '', '']
#sofa
supps[17] = ['', '', '', '', '', '', '', '', '']
#train
supps[18] = ['', '', '', '', '', '', '', '', '']
#tvmonitor
supps[19] = ['', '', '', '', '', '', '', '', '']

class Pascal5FewShotDset(torch_data.Dataset):
    """This data set capsules the pascal-5i dataset and implements few shot segmentation requirements"""
    def __init__(self, root, train, idx,
                 transform, tgt_transform, supp_transform, shuffle_lasses=False, shots=None, maskSupps=False):
        self.classes = np.array(range(idx * 5, (idx * 5) + 5))
        # self.classes = np.delete(self.classes, np.where(self.classes == 2))
        # self.classes = np.array([2])
        self.shuffleClasses = shuffle_lasses
        self.shots = shots
        self.idx = idx
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.tgt_transform = tgt_transform
        self.supp_transform = supp_transform
        self.maskSupps = maskSupps
        self.length = 0
        self.unUsedIndices = {}
        if train:
            self.data, self.suppSets = self.load_all_data(root, "train", idx)
        else:
            self.data, self.suppSets = self.load_all_data(root, "test", idx)

        for key, dset in self.data.items():
            self.unUsedIndices[key] = list(range(dset.shape[0]))
            self.length += dset.shape[0]

    def __getitem__(self, index):
        """This returns self.shots support sets. If self.shots is not defined a random number
        between 1 and 4 support sets are returned together with the query image and the query ground truth segmentaiton"""
        if self.classes.size == 0: # be ready for next epoch
            self.classes = np.array(range(self.idx * 5, (self.idx * 5) + 5))
            # self.classes = np.delete(self.classes, np.where(self.classes == 2))
            # self.classes = np.array([2])
            for key, dset in self.data.items():
                self.unUsedIndices[key] = list(range(dset.shape[0]))

        if self.shots is None:
            shots = np.random.randint(1, 4)
        else:
            shots = self.shots

        if self.shuffleClasses:
            cl = np.random.choice(self.classes)
        else:
            cl = self.classes[0]
        dset = self.data[cl]
        supset = self.suppSets[cl]

        s_sample_idx = np.random.choice(range(len(supset)), (shots,), replace=False)
        q_sample_idx = np.random.choice(self.unUsedIndices[cl])

        # be sure to use a query image only once
        self.unUsedIndices[cl].remove(q_sample_idx)
        if self.unUsedIndices[cl] == []:
            self.classes = np.delete(self.classes, np.where(self.classes == cl))

        # draw images from data set and apply transforms
        s_sets = None
        for shot in range(shots):
            idx = s_sample_idx[shot]
            supp_sample = supset[idx]
            supp_o = Image.fromarray(supp_sample[0], mode='RGB')
            supp_g = Image.fromarray(supp_sample[1], mode='RGB')
            if self.transform is not None:
                supp_o = self.supp_transform(supp_o)
                supp_g = self.tgt_transform(supp_g)
            if self.maskSupps:
                s_set = supp_o * supp_g
            else:
                s_set = torch.cat([supp_o, supp_g.unsqueeze(0).float()], dim=0)
            s_set = s_set.unsqueeze(0)
            if s_sets is None:
                s_sets = s_set
            else:
                s_sets =torch.cat([s_sets, s_set], dim=0)

        query_sample = dset[q_sample_idx]
        q_img = Image.fromarray(query_sample[0], mode='RGB')
        q_gt = Image.fromarray(query_sample[1], mode='RGB')
        if self.transform is not None:
            q_img = self.transform(q_img)
            q_gt = self.tgt_transform(q_gt)
        q_gt = q_gt.unsqueeze(0)  # no binary class seg. Need to round
        q_gt = torch.cat((q_gt==0.0, q_gt==1.0) ,0).long()  # need a two channel gt for fore- and background

        return (s_sets, q_img), q_gt

    def __len__(self):
        return self.length

    def load_all_data(self, dir, kind, idx):
        """Load the data set into ram"""
        # Ideas: optionally train and test without conditioner. Train segmenter with various sized few shots and with all idices.
        load_dir = os.path.join(dir, config.root_dir, str(idx), kind)
        test_dir = os.path.join(dir, config.root_dir, str(idx), 'test')
        all_data = {}
        all_sups = {}
        for cl in self.classes:
            classFile = open(os.path.join(load_dir, config.class_list[cl]+".txt"))
            content = classFile.readlines()
            content = [x.strip() for x in content]
            classFile.close()
            data = []
            for c in content:
                fo = Image.open(os.path.join(load_dir, "groundtruth", c + ".jpg"))
                g = np.asarray(fo)
                fo.close()
                fo = Image.open(os.path.join(load_dir, "origin", c + ".jpg"))
                o = np.asarray(fo)
                fo.close()
                data.append((o, g))
            all_data[cl] = np.array(data)
            # load all supp sets:
            suppSets = []
            for c in supps[cl]:
                fo = Image.open(os.path.join(test_dir, "groundtruth", c + ".jpg"))
                g = np.asarray(fo)
                fo.close()
                fo = Image.open(os.path.join(test_dir, "origin", c + ".jpg"))
                o = np.asarray(fo)
                fo.close()
                suppSets.append((o, g))
            all_sups[cl] = suppSets

        return all_data, all_sups
