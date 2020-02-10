import torch
from models.GCNNs.cstm_message_passing import GcnEdgeConv, EdgeConv, NodeConv, EdgeConvNoEdge
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch.nn as nn
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
        x = self.e_conv4(x, edge_index)
        # x, e = F.dropout(x, training=self.training), F.dropout(e, training=self.training)
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
        self.e_conv1 = GATConv(n_node_features_in, 5)
        self.e_conv2 = GATConv(5, 10)
        self.e_conv3 = GATConv(10, 10)
        self.e_conv7 = EdgeConvNoEdge(10, 10)
        self.e_conv9 = torch.nn.Linear(10, n_edge_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        self.loss = torch.nn.L1Loss()
        self.device = device

    def forward(self, x, e, edge_index):
        x = self.e_conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.e_conv3(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        e = self.e_conv7(x, edge_index)
        e = F.dropout(e, training=self.training)
        e = F.relu(e)
        e = self.e_conv9(e)
        e = F.dropout(e, training=self.training)
        e = F.relu(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e


class GcnEdgeConvNet4(torch.nn.Module):
    def __init__(self, n_node_features_in, n_edge_features_in, n_edge_classes, device, softmax=True):
        super(GcnEdgeConvNet4, self).__init__()
        self.softmax = softmax
        self.e_conv1 = GATConv(n_node_features_in, 5)
        self.e_conv2 = GATConv(5, 10)
        self.e_conv3 = GATConv(10, 10)
        self.e_conv7 = EdgeConv(10, n_edge_features_in, 10)
        self.e_conv9 = torch.nn.Linear(10, n_edge_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        self.loss = torch.nn.L1Loss()
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
        e = self.e_conv7(x, e, edge_index)
        # e = F.dropout(e, training=self.training)
        e = F.relu(e)
        e = self.e_conv9(e)
        # e = F.dropout(e, training=self.training)
        e = F.relu(e)
        if self.softmax:
            return nn.functional.softmax(e, -1)
        return e