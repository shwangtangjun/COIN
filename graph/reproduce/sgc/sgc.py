import torch
import torch.nn as nn
from time import perf_counter


# preprocessing stage
def sgc_precompute(features, adj, degree=2):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time


# classification stage
class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    Same as SGC's classification head
    """

    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)