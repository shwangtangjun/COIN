import math
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F



class GCNLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super().__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_u(u='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GDEFunc(nn.Module):
    def __init__(self, gnn):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        self.nfe = 0

    def set_graph(self, g: dgl.DGLGraph):
        for layer in self.gnn:
            layer.g = g

    def forward(self, t, x):
        self.nfe += 1
        x = self.gnn(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='dopri5', rtol=1e-3, atol=1e-4, adjoint=True):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol

    def forward(self, x: torch.Tensor, T: int = 1):
        self.integration_time = torch.tensor([0, T]).float()
        self.integration_time = self.integration_time.type_as(x)

        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        return out[-1]


def GCDE(g, in_features, out_features):
    gnn = nn.Sequential(GCNLayer(g=g, in_feats=64, out_feats=64, activation=nn.Softplus(), dropout=0.9),
                        GCNLayer(g=g, in_feats=64, out_feats=64, activation=None, dropout=0.9)
                        )
    gdefunc = GDEFunc(gnn)
    gde = ODEBlock(odefunc=gdefunc, method='rk4', atol=1e-3, rtol=1e-4, adjoint=False)

    model = nn.Sequential(GCNLayer(g=g, in_feats=in_features, out_feats=64, activation=F.relu, dropout=0.4),
                          gde,
                          GCNLayer(g=g, in_feats=64, out_feats=out_features, activation=None, dropout=0.))
    return model
