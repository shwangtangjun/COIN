import torch.nn as nn
import torch.nn.functional as F


class SimpleResNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleResNet, self).__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, features):
        features = features + F.relu(self.fc(features))
        out = self.classifier(features)
        out = F.softmax(out, dim=1)
        return out


class COIN(nn.Module):
    def __init__(self, in_features, layer_num, sigma2):
        super(COIN, self).__init__()
        self.sigma2 = sigma2
        self.layer_num = layer_num

        self.base_classifier = SimpleResNet(in_features, 5)  # 5-way classification

    def forward(self, features, laplacian):
        u = self.base_classifier(features)
        for i in range(self.layer_num):
            u = u - self.sigma2 * laplacian.mm(u)
        return u
