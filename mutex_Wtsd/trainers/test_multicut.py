import numpy as np
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from data.disc_datasets import DiscSpGraphDset
from models.simple_unet import UNet
from torch.utils.data import DataLoader
from utils.general import calculate_gt_edge_costs
import torch
import matplotlib.pyplot as plt
assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.FloatTensor)

separating_channel = 2
offsets = [[0, -1], [-1, 0],
           [-3, 0], [0, -3]]
rootPath = '/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/models'
modelFileCircle = os.path.join(rootPath, 'UnetEdgePreds.pth')

affinities_predictor_circle = UNet(n_channels=1, n_classes=len(offsets), bilinear=True, device=device)
affinities_predictor_circle.load_state_dict(torch.load(modelFileCircle), strict=True)
affinities_predictor_circle.cuda()
dloader_disc = DataLoader(DiscSpGraphDset(affinities_predictor_circle, separating_channel, offsets), batch_size=1,
                          shuffle=True, pin_memory=True)
neighbors, nodes, seg, gt_seg, affs, gt_affs = next(iter(dloader_disc))

offsets = [[0, 0, -1], [0, -1, 0],
           [0, -3, 0], [0, 0, -3]]

affs = np.transpose(affs.cpu().numpy(), (1, 0, 2, 3))
gt_affs = np.transpose(gt_affs.cpu().numpy(), (1, 0, 2, 3))
seg = seg.cpu().numpy()
gt_seg = gt_seg.cpu().numpy()
boundary_input = np.mean(affs, axis=0)
gt_boundary_input = np.mean(gt_affs, axis=0)

rag = feats.compute_rag(seg)
# edges rag.uvIds() [[1, 2], ...]

costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
gt_costs = calculate_gt_edge_costs(rag.uvIds(), seg.squeeze(), gt_seg.squeeze())

edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
gt_edge_sizes = feats.compute_boundary_mean_and_length(rag, gt_boundary_input)[:, 1]
costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
gt_costs = mc.transform_probabilities_to_costs(gt_costs, edge_sizes=edge_sizes)

node_labels = mc.multicut_kernighan_lin(rag, costs)
gt_node_labels = mc.multicut_kernighan_lin(rag, gt_costs)

segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
gt_segmentation = feats.project_node_labels_to_pixels(rag, gt_node_labels)
plt.imshow(np.concatenate((gt_segmentation.squeeze(), segmentation.squeeze(), seg.squeeze()), axis=1));plt.show()
