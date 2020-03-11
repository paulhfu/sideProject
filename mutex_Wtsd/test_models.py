import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# from models.GCNNs.mc_pooling_1 import test
from data.datasets import DiscSpGraphDset
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeConvNet3
from models.simple_unet import UNet
from torch.utils.data import DataLoader
import numpy as np
import torch

assert torch.cuda.device_count() == 1
torch.set_default_tensor_type('torch.FloatTensor')
# Detect if we have a GPU available
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
model = GcnEdgeConvNet3(n_node_features_in=1, n_edge_classes=1, device=device, softmax=False)
# model.load_state_dict(torch.load('graph_conv_circle'), strict=True)
model.cuda(device=model.device)
model.train()
# check_no_singles(edges.numpy(), len(node_feat))
# check if input is nan
# tst = np.sum(torch.cat((node_feat.view(-1), edge_feat.view(-1), gt_edge_weights.view(-1))).numpy())
# if np.isnan(tst) or np.isinf(tst):
#     print("some values from input are nan")
# edges, edge_feat, gt_edge_weights, node_feat, seg, gt_seg, affinities, _ = next(iter(dloader_disc))
# affs = np.expand_dims(affinities.squeeze().numpy(), axis=1)
# boundary_input = np.mean(affs, axis=0)
# mc_seg = utils.multicut_from_probas(seg.squeeze().numpy(), edges.squeeze().t().contiguous().numpy(), edge_feat.squeeze().numpy(), boundary_input)
# gt_mc_seg = utils.multicut_from_probas(seg.squeeze().numpy(), edges.squeeze().t().contiguous().numpy(), gt_edge_weights.squeeze(), boundary_input)
# mc_seg = cm.prism(mc_seg / mc_seg.max())
# seg = cm.prism(seg.squeeze().numpy() / seg.squeeze().numpy().max())
# gt_mc_seg = cm.prism(gt_mc_seg / gt_mc_seg.max())
# plt.imshow(np.concatenate((mc_seg, gt_mc_seg, seg)));plt.show()


with torch.autograd.set_grad_enabled(True):
    for epoch in range(10000):
        edges, edge_feat, gt_edge_weights, node_feat, seg, gt_seg, affinities, _ = next(iter(dloader_disc))
        node_feat, edge_feat, edges, gt_edge_weights = node_feat.squeeze(0), edge_feat.squeeze(0), edges.squeeze(
            0), gt_edge_weights.squeeze(0)
        node_feat = node_feat.to(model.device)
        edge_feat = edge_feat.to(model.device)
        edges = edges.to(model.device)
        gt_edge_weights = gt_edge_weights.to(model.device)

        # assert all((node_feat.squeeze()[edges.t()][:50, 1] != node_feat.squeeze()[edges.t()][:50, 0]).float() == gt_edge_weights)

        model.optimizer.zero_grad()
        out, tt = model(node_feat, edges)
        assert model.loss(tt.squeeze(), gt_edge_weights).item() == 0
        # loss = model.loss(out, torch.stack(((gt_edge_weights == 0), (gt_edge_weights == 1)), dim=-1).float())
        loss = model.loss(out.squeeze(), gt_edge_weights)
        # loss1 = model.loss(out.squeeze(), edge_feat.squeeze()).detach()
        loss.backward()
        print('loss: ', loss.item())
        # print('loss: ', loss1.item())
        model.optimizer.step()
affs = np.expand_dims(affinities.squeeze().numpy(), axis=1)
# boundary_input = np.mean(affs, axis=0)
# mc_seg1 = utils.multicut_from_probas(seg.squeeze().numpy(), edges.squeeze().cpu().t().contiguous().numpy(), edge_feat.squeeze().cpu().numpy(), boundary_input)
# mc_seg = utils.multicut_from_probas(seg.squeeze().numpy(), edges.squeeze().cpu().t().contiguous().numpy(), out.detach().squeeze().cpu().numpy(), boundary_input)
# gt_mc_seg = utils.multicut_from_probas(seg.squeeze().numpy(), edges.squeeze().cpu().t().contiguous().numpy(), gt_edge_weights.squeeze(), boundary_input)
# mc_seg = cm.prism(mc_seg / mc_seg.max())
# mc_seg1 = cm.prism(mc_seg1 / mc_seg1.max())
# seg = cm.prism(seg.squeeze().numpy() / seg.squeeze().numpy().max())
# gt_mc_seg = cm.prism(gt_mc_seg / gt_mc_seg.max())
# plt.imshow(np.concatenate((mc_seg1, mc_seg, gt_mc_seg, seg)));plt.show()
# torch.save(model.state_dict(), 'graph_conv_circle')