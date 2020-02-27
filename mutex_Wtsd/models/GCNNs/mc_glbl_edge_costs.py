import torch
from models.GCNNs.cstm_message_passing import GcnEdgeConv, EdgeConv, NodeConv, EdgeConvNoEdge, SpatEdgeConv, SpatNodeConv
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv


class GcnEdgeConvNet(torch.nn.Module):
    def __init__(self, n_node_features_in, n_edge_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeConvNet, self).__init__()
        self.softmax = True
        self.e_conv1 = GCNConv(n_node_features_in, 15)
        self.e_conv2 = GCNConv(15, 25)
        self.e_conv3 = GCNConv(25, 30)
        self.e_conv4 = GcnEdgeConv(30, n_edge_features_in, 35, 15)
        self.e_conv5 = GcnEdgeConv(35, 15, 40, 20)
        self.e_conv6 = GcnEdgeConv(40, 20, 45, 25)
        self.e_conv7 = EdgeConv(45, 25, n_edge_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, x, e, edge_index):
        x = self.e_conv1(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv3(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x, e = self.e_conv4(x, e, edge_index)
        # x, e = F.dropout(x, training=self.training), F.dropout(e, training=self.training)
        x, e = F.relu(x), F.relu(e)
        x, e = self.e_conv5(x, e, edge_index)
        # x, e = F.dropout(x, training=self.training), F.dropout(e, training=self.training)
        x, e = F.relu(x), F.relu(e)
        x, e = self.e_conv6(x, e, edge_index)
        # x, e = F.dropout(x, training=self.training), F.dropout(e, training=self.training)
        x, e = F.relu(x), F.relu(e)
        e = self.e_conv7(x, e, edge_index)
        e = F.relu(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e


class GcnEdgeConvNet2(torch.nn.Module):
    def __init__(self, n_node_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeConvNet2, self).__init__()
        self.softmax = True
        self.e_conv1 = NodeConv(n_node_features_in, 15)
        self.e_conv2 = NodeConv(15, 25)
        self.e_conv3 = NodeConv(25, 30)
        self.e_conv4 = NodeConv(30, 30)
        self.e_conv5 = NodeConv(30, 40)
        self.e_conv6 = NodeConv(40, 40)
        self.e_conv7 = EdgeConvNoEdge(40, 40)
        self.e_conv8 = torch.nn.Linear(40, 40)
        self.e_conv9 = torch.nn.Linear(40, n_edge_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, x, e, edge_index):
        x = self.e_conv1(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv3(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv4(x, edge_index)
        # x, e = F.dropout(x, traself.e_conv9 = torch.nn.Linear(n_node_features_in, n_edge_classes)ining=self.training), F.dropout(e, training=self.training)
        x = F.relu(x)
        x = self.e_conv5(x, edge_index)
        # x, e = F.dropout(x, training=self.training), F.dropout(e, training=self.training)
        x, e = F.relu(x), F.relu(e)
        x = self.e_conv6(x, edge_index)
        # x, e = F.dropout(x, training=self.training), F.dropout(e, training=self.training)
        x = F.relu(x)
        e = self.e_conv7(x, edge_index)
        e = F.relu(e)
        e = self.e_conv8(e)
        e = F.relu(e)
        e = self.e_conv9(e)
        e = F.relu(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e


class GcnEdgeConvNet3(torch.nn.Module):
    def __init__(self, n_node_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeConvNet3, self).__init__()
        self.softmax = softmax
        # self.e_conv1 = GATConv(n_node_features_in, 5)
        # self.e_conv2 = GATConv(5, 10)
        # self.e_conv3 = GATConv(10, 10)
        self.e_conv7 = EdgeConvNoEdge(n_node_features_in, 20)
        self.e_conv8 = torch.nn.Linear(20, 10)
        self.e_conv81 = torch.nn.Linear(10, 10)
        self.e_conv82 = torch.nn.Linear(10, 5)
        self.e_conv9 = torch.nn.Linear(5, n_edge_classes)
        self.nl = torch.nn.LeakyReLU(negative_slope=0.1)
        self.bn = BatchNorm1d(n_node_features_in)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = torch.nn.MSELoss()
        self.device = device

    def forward(self, x, edge_index):
        # x = self.bn(x)
        # x = self.e_conv1(x, edge_index)
        # x = F.dropout(x, training=self.training)
        # x = self.nl(x)
        # x = self.e_conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        # x = self.nl(x)
        # x = self.e_conv3(x, edge_index)
        # # x = F.dropout(x, training=self.training)
        # x = self.nl(x)
        e, tt = self.e_conv7(x, edge_index)
        # e = F.dropout(e, training=self.training)
        e = self.nl(e)
        e = self.e_conv8(e)
        e = self.nl(e)
        e = self.e_conv81(e)
        e = self.nl(e)
        e = self.e_conv82(e)
        e = self.nl(e)
        e = self.e_conv9(e)
        # # e = F.dropout(e, training=self.training)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        e = torch.nn.functional.sigmoid(e)
        return e, tt


class GcnEdgeConvNet4(torch.nn.Module):
    def __init__(self, n_node_features_in, n_edge_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeConvNet4, self).__init__()
        self.softmax = softmax
        self.e_conv7 = EdgeConv(n_node_features_in, n_edge_features_in, 6)
        self.e_conv8 = torch.nn.Linear(6, 12)
        self.e_conv81 = torch.nn.Linear(12, 6)
        # self.e_conv82 = torch.nn.Linear(256, 128)
        # self.e_conv83 = torch.nn.Linear(128, 64)
        # self.e_conv84 = torch.nn.Linear(64, 16)
        self.e_conv9 = torch.nn.Linear(6, n_edge_classes)
        self.nl = torch.nn.LeakyReLU(negative_slope=0.1)
        self.bn = BatchNorm1d(n_node_features_in)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()
        self.device = device

    def forward(self, x, e, edge_index):
        # x = self.bn(x)
        # e = self.bn(e)
        e, tt = self.e_conv7(x, e, edge_index)
        e = self.nl(e)
        e = self.e_conv8(e)
        e = self.nl(e)
        e = self.e_conv81(e)
        e = self.nl(e)
        # e = self.e_conv82(e)
        # e = self.nl(e)
        # e = self.e_conv83(e)
        # e = self.nl(e)
        # e = self.e_conv84(e)
        # e = self.nl(e)
        e = self.e_conv9(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e, tt


class GcnEdgeConvNet5(torch.nn.Module):
    def __init__(self, n_node_features_in, n_edge_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeConvNet5, self).__init__()
        self.softmax = softmax
        self.e_conv7 = EdgeConv(n_node_features_in, n_edge_features_in, 6)
        self.e_conv8 = torch.nn.Linear(6, 12)
        self.e_conv81 = torch.nn.Linear(12, 6)
        # self.e_conv82 = torch.nn.Linear(256, 128)
        # self.e_conv83 = torch.nn.Linear(128, 64)
        # self.e_conv84 = torch.nn.Linear(64, 16)
        self.e_conv9 = torch.nn.Linear(6, n_edge_classes)
        self.nl = torch.nn.LeakyReLU(negative_slope=0.1)
        self.bn = BatchNorm1d(n_node_features_in)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()
        self.device = device

    def forward(self, x, e, edge_index):
        # x = self.bn(x)
        # e = self.bn(e)
        e, tt = self.e_conv7(x, e, edge_index)
        e = self.nl(e)
        e = self.e_conv8(e)
        e = self.nl(e)
        e = self.e_conv81(e)
        e = self.nl(e)
        # e = self.e_conv82(e)
        # e = self.nl(e)
        # e = self.e_conv83(e)
        # e = self.nl(e)
        # e = self.e_conv84(e)
        # e = self.nl(e)
        e = self.e_conv9(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e, tt


class GcnEdgeAngleConv1(torch.nn.Module):
    def __init__(self, n_node_channels_in, n_edge_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeAngleConv1, self).__init__()
        self.softmax = softmax
        self.node_conv1 = SpatNodeConv(n_node_channels_in, 64)
        self.edge_conv1 = SpatEdgeConv(64, 128)
        self.node_conv2 = SpatNodeConv(64, 128)
        self.edge_conv2 = SpatEdgeConv(128, 128, use_init_edge_feats=True, n_channels_in=128)
        self.global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.out_lcf1 = nn.Linear(128 + n_edge_features_in + 1, 256)
        self.out_lcf2 = nn.Linear(256, n_edge_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()
        self.device = device

    def forward(self, node_features, edge_features_1d, edge_index, angles, edge_weights):
        edge_weights = torch.cat((edge_weights, edge_weights), dim=0)
        node_features, _ = self.node_conv1(node_features, edge_index, angles, edge_weights)
        _, edge_features = self.edge_conv1(node_features, edge_index, edge_weights)
        node_features, _ = self.node_conv2(node_features, edge_index, angles, edge_weights)
        _, edge_features = self.edge_conv2(node_features, edge_index, edge_weights, edge_features)

        pooled_features = self.global_pool(edge_features)
        e = self.out_lcf1(torch.cat((pooled_features, edge_features_1d, edge_weights), 0))
        e = self.out_lcf2(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e

