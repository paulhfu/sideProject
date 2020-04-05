import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from data.datasets import DiscSpGraphDset
from utils import general
from models.simple_unet import UNet
import numpy as np
from tqdm import tqdm

assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.FloatTensor)

strides = None
separating_channel = 2
offsets = [[0, -1], [-1, 0],
           [-3, 0], [0, -3]]

ndf = 4
nc = 1

class AE(nn.Module):
    # https://github.com/coolvision/vae_conv/blob/master/mvae_conv_model.py
    def __init__(self, device):
        super(AE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 6, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 6, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 10, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 10, ndf * 12, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 12, ndf * 14, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ndf * 14, ndf * 12, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 12),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 12, ndf * 10, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 10, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 8, ndf * 6, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 6),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 6, ndf * 4, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x);
        return conv

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, feature_size):
        z = self.encode(x)
        bbox = []
        mask = torch.ones(size=z.shape).to(self.device)
        for sh, sz in zip(z.shape[-2:], feature_size):
            diff = (sh - sz) / 2
            bbox.append((int(diff), int(sh - diff)))

        mask[:, :, :bbox[0][0], :] = 0
        mask[:, :, bbox[0][1]:, :] = 0
        mask[:, :, :, :bbox[1][0]] = 0
        mask[:, :, :, bbox[1][1]:] = 0

        z = z * mask
        decoded = self.decode(z)
        return decoded, z

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    # BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), size_average=False)
    return F.mse_loss(recon_x, x)

def get_boxed_sp(nodes, segmentation, raw, size):
    haloed_seg = torch.zeros(size=(segmentation.shape[0] + size[0] * 2, segmentation.shape[1] + size[1] * 2))
    haloed_raw = torch.zeros(size=(segmentation.shape[0] + size[0] * 2, segmentation.shape[1] + size[1] * 2))
    haloed_seg[size[0]:haloed_seg.shape[0]-size[0], size[1]:haloed_seg.shape[1]-size[1]] = segmentation
    haloed_raw[size[0]:haloed_seg.shape[0]-size[0], size[1]:haloed_seg.shape[1]-size[1]] = raw
    sp_boxes = []
    for i, n in enumerate(nodes):
        mask = (n == haloed_seg)
        y, x = general.bbox(mask.unsqueeze(0).numpy())
        x, y = x[0], y[0]
        for co, sz in zip((y, x), size):
            diff = sz - (co[1] - co[0])
            if diff > 0:
                if diff % 2 != 0:
                    diff += 1
                one_sided_halo = diff/2
                co[0] -= one_sided_halo
                co[1] += one_sided_halo
            if (co[1] - co[0]) % 2 != 0:
                co[1] += 1
            co[0], co[1] = int(co[0]), int(co[1])
        masked_seg = mask.float() * haloed_raw
        sp_boxes.append(masked_seg[y[0]:y[1], x[0]:x[1]])
    return sp_boxes

def train(sp_boxes, model, optimizer, feature_size):
    model.train()
    train_loss = 0
    for batch_idx, sp in enumerate(sp_boxes):
        sp = sp.to(device).unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()
        recon_sp, z = model(sp, feature_size)
        loss = loss_function(recon_sp, sp)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print('loss', loss.item())
    return train_loss

def main():
    model = AE(device)
    model.cuda(device)
    feature_size = [34, 34]
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    file = 'mask/masks.h5'
    rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/models'
    modelFileCircle = os.path.join(rootPath, 'UnetEdgePreds.pth')
    affinities_predictor_circle = UNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
    affinities_predictor_circle.load_state_dict(torch.load(modelFileCircle), strict=True)
    affinities_predictor_circle.cuda(device)
    dloader = DataLoader(DiscSpGraphDset(affinities_predictor_circle, separating_channel, offsets), batch_size=1,
                              shuffle=True, pin_memory=True)

    writer = SummaryWriter(logdir='./logs')

    for e in tqdm(range(10000)):
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_feat, seg, raw, affinities, _, angles = next(iter(dloader))
        seg = seg.squeeze()
        raw = raw.squeeze()
        seg += 1
        sp_boxes = get_boxed_sp(np.unique(seg), seg, raw, size=feature_size)
        seg -= 1

        loss = train(sp_boxes, model, optimizer, feature_size)

        writer.add_scalar("loss/sp_ae", loss, e)

    torch.save(model.state_dict(), 'ae_sp_feat_model.pth')
