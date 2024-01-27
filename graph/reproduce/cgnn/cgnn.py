import torch
from torch import nn
import torch.nn.functional as F

adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):
    # currently requires in_features = out_features
    def __init__(self, in_features, out_features, opt, adj):
        super(ODEFunc, self).__init__()
        self.opt = opt
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))

        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)

    def forward(self, t, x):
        self.nfe += 1

        alph = torch.sigmoid(self.alpha_train).unsqueeze(dim=1)
        ax = torch.spmm(self.adj, x)
        f = alph * 0.5 * (ax - x) + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.nfe = 0

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        self.nfe += 1

        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t)[1]
        return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
            + ")"


# Define the GNN model.
class CGNN(nn.Module):
    def __init__(self, opt, adj):
        super(CGNN, self).__init__()
        self.opt = opt
        self.m1 = nn.Linear(opt['num_feature'], opt['hidden_dim'])

        self.odeblock = ODEblock(ODEFunc(2 * opt['hidden_dim'], 2 * opt['hidden_dim'], opt, adj),
                                 t=torch.tensor([0, opt['time']]))

        self.m2 = nn.Linear(opt['hidden_dim'], opt['num_class'])

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        # Encode each node based on its feature.
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        # Solve the initial value problem of the ODE.
        c_aux = torch.zeros(x.shape).to(x.device)
        x = torch.cat([x, c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1] // 2, dim=1)[0]

        # Activation.
        z = F.relu(z)

        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z
