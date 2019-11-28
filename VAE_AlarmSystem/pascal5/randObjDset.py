from .config import config
import torch.utils.data as torch_data
from PIL import Image
import torch
import numpy as np
import random
import os

class RandObjDset(torch_data.Dataset):
    """This data set capsules the random objects dataset and implements few shot segmentation requirements"""
    def __init__(self, root, transform, tgt_transform, supp_transform, shots=None, maskSupps=False):
        self.shots = shots
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.tgt_transform = tgt_transform
        self.supp_transform = supp_transform
        self.maskSupps = maskSupps

        self.length = 800
        self.data, self.suppSets = self.load_all_data(root)

    def __getitem__(self, index):
        s_sets = None
        supp_sample = self.suppSets[index]
        s_sample_idx = np.random.choice(range(5), (self.shots,), replace=False)
        for shot in range(self.shots):
            idx = s_sample_idx[shot]
            supp_o = Image.fromarray(supp_sample[idx][0])
            supp_g = Image.fromarray(supp_sample[idx][1])
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            if self.transform is not None:
                random.seed(seed)  # apply this seed to img tranfsorms
                supp_o = self.supp_transform(supp_o)
                random.seed(seed)  # apply this seed to img tranfsorms
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

        query_sample = self.data[index]
        q_o = Image.fromarray(query_sample[0], mode='RGB')
        q_g= Image.fromarray(query_sample[1], mode='RGB')
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        if self.transform is not None:
            random.seed(seed)
            q_o = self.transform(q_o)
            random.seed(seed)
            q_g = self.tgt_transform(q_g)
        q_g = q_g.unsqueeze(0)  # no binary class seg. Need to round
        q_g = torch.cat((q_g==0.0, q_g==1.0) ,0).long()  # need a two channel gt for fore- and background

        return (s_sets, q_o), q_g

    def __len__(self):
        return self.length

    def load_all_data(self, dir):
        """Load the data set into ram"""
        # Ideas: optionally train and test without conditioner. Train segmenter with various sized few shots and with all idices.
        load_dir = os.path.join(dir, config.root_dir, 'randomObjects')
        data = {}
        supps = {}
        for i in range(1, self.length + 1):
            supppset = []
            fo = Image.open(os.path.join(load_dir, "query", "groundtruth", str(i) + ".jpg"))
            g = np.asarray(fo)
            fo.close()
            fo = Image.open(os.path.join(load_dir, "query", "origin", str(i) + ".jpg"))
            o = np.asarray(fo)
            fo.close()
            data[i-1] = np.array((o, g))

            for j in range(1, 6):
                fo = Image.open(os.path.join(load_dir, "support", "groundtruth", str(i), str(i) + str(j) + ".jpg"))
                g = np.asarray(fo)
                fo.close()
                fo = Image.open(os.path.join(load_dir, "support", "origin", str(i), str(i) + str(j) + ".jpg"))
                o = np.asarray(fo)
                fo.close()
                supppset.append((o, g))
            supps[i-1] = np.array(supppset)

        return data, supps