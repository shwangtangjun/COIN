import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, num_classes, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support1 = torch.mm(input, self.weight1)
        output1 = torch.spmm(adj, support1)

        support2 = torch.mm(input, self.weight2)
        output2 = support2

        output = output1 + output2

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
