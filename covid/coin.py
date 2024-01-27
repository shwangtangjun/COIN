import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import DCRNN, MPNNLSTM, GConvGRU, A3TGCN, EvolveGCNH, TGCN


class SimpleResNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super(SimpleResNet, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_features)
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, features):
        out = self.shortcut(features) + self.linear2(F.elu(self.linear1(features)))
        return out


class COIN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, layer_num, sigma2, dropout):
        super(COIN, self).__init__()

        self.base_classifier = SimpleResNet(in_features, out_features, hidden_dim)

        self.layer_num = layer_num
        self.sigma2 = sigma2
        self.dropout = dropout

    def forward(self, features, laplacian):
        u = self.base_classifier(features)

        for i in range(self.layer_num):
            u = F.dropout(u, p=self.dropout, training=self.training)
            u = u - self.sigma2 * laplacian.mm(u)

        return u


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


# Below implementations are collected from PyTorch Geometric Temporal Examples
class DCRNNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(DCRNNNet, self).__init__()
        self.recurrent = DCRNN(in_channels, hidden_channels, K=1)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.elu(h)
        h = self.linear(h)
        return h


class MPNNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(MPNNNet, self).__init__()
        self.recurrent = MPNNLSTM(in_channels, hidden_channels, 129, window=1, dropout=dropout)
        self.linear = nn.Linear(2 * hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.elu(h)
        h = self.linear(h)
        return h


class GConvGRUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GConvGRUNet, self).__init__()
        self.recurrent = GConvGRU(in_channels, hidden_channels, K=1)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.elu(h)
        h = self.linear(h)
        return h


class A3TGCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(A3TGCNNet, self).__init__()
        self.recurrent = A3TGCN(1, hidden_channels, periods=in_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.elu(h)
        h = self.linear(h)
        return h


class EvolveGCNNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(EvolveGCNNet, self).__init__()
        self.recurrent = EvolveGCNH(129, in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.elu(h)
        h = self.linear(h)
        return h
