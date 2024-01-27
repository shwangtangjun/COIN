import numpy as np
import torch
import sys
sys.path.append("../..")
from data_process.make_dataset import get_dataset, get_train_val_test_split
from torch_geometric.data import Data


def load_data(dataset_str, random_state):
    data_path = "../../data/" + dataset_str + ".npz"

    adj, features, labels = get_dataset(dataset_str, data_path, standardize=True, train_examples_per_class=20,
                                        val_examples_per_class=30)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=20,
                                                            val_examples_per_class=30, test_size=None)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels.argmax(axis=-1))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def fill_missing_keys(opt):
    if 'max_test_steps' not in opt:
        opt['max_test_steps'] = 100

    if 'earlystopxT' not in opt:
        opt['earlystopxT'] = 3

    return opt


class torch_geometric_like_dataset():
    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.num_classes = labels.max().item() + 1
        self.num_nodes = features.shape[0]
        self.num_features = features.shape[1]
        edge_index = adj.to_dense().nonzero().t().contiguous()
        self.data = Data(x=features, y=labels, edge_index=edge_index)

        self.data.train_mask = self.create_mask(idx_train)
        self.data.val_mask = self.create_mask(idx_val)
        self.data.test_mask = self.create_mask(idx_test)

    def create_mask(self, idx):
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask
