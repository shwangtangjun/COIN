import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(ResNet, self).__init__()

        assert in_features % 3 == 0

        self.weight = nn.Parameter(torch.FloatTensor(1, in_features))
        self.bias = nn.Parameter(torch.FloatTensor(in_features // 3))

        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

        self.fc1 = nn.Linear(in_features // 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.shortcut = nn.Linear(in_features // 3, out_features)

    def forward(self, features, return_features=False):
        features = features * self.weight
        features = features.reshape(features.shape[0], -1, 3)
        out = features.sum(dim=2)
        out = out + self.bias

        if return_features:
            return out
        out = self.shortcut(out) + self.fc2(F.relu(self.fc1(out)))
        return torch.sigmoid(out)


class COIN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, layer_num, sigma2):
        super(COIN, self).__init__()

        self.base_classifier = ResNet(in_features, hidden_dim, out_features)

        self.layer_num = layer_num
        self.sigma2 = sigma2

    def forward(self, features, laplacian):
        u = self.base_classifier(features)

        for i in range(self.layer_num):
            u = u - self.sigma2 * laplacian.mm(u)

        return u
