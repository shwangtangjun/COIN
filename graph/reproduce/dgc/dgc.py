import torch
import torch.nn as nn
from time import perf_counter
import scipy.sparse as sp
from utils_dgc import sparse_mx_to_torch_sparse_tensor


# preprocessing stage
def dgc_precompute(features, adj, T, K):
    # integration with the forward Euler scheme by default
    if K == 0 or T == 0.:
        return features, 0.
    delta = T / K
    t = perf_counter()

    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)

    op = (1 - delta) * eye + delta * adj
    for i in range(K):
        features = torch.spmm(op, features)
    precompute_time = perf_counter() - t
    return features, precompute_time


# classification stage
class DGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    Same as SGC's classification head
    """

    def __init__(self, nfeat, nclass):
        super(DGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
