import numpy as np
import scipy.sparse as sp
import torch
import sys
sys.path.append("../..")
from data_process.make_dataset import get_dataset, get_train_val_test_split


def load_data(dataset_str, random_state, self_loop_weight):
    data_path = "../../data/" + dataset_str + ".npz"

    adj, features, labels = get_dataset(dataset_str, data_path, standardize=True, train_examples_per_class=20,
                                        val_examples_per_class=30)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=20,
                                                            val_examples_per_class=30, test_size=None)

    features = normalize_features(features)
    adj = normalize_adj(adj + self_loop_weight * sp.eye(adj.shape[0]), normalization="symmetric")

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels.argmax(axis=-1))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj, normalization="symmetric"):
    """Symmetrically or row normalize adjacency matrix."""
    if normalization == "symmetric":
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        mx = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    elif normalization == "row":
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(adj)
    else:
        raise NotImplementedError
    return mx


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


def get_best_params(dataset):
    opt = dict()

    if dataset == 'cora':
        opt['hidden_dim'] = 16
        opt['input_dropout'] = 0.5
        opt['dropout'] = 0
        opt['optimizer'] = 'rmsprop'
        opt['decay'] = 5e-4
        opt['lr'] = 0.00147
        opt['self_loop_weight'] = 0.595
        opt['alpha'] = 0.885
        opt['time'] = 23.9
        opt['num_feature'] = 1433
        opt['num_class'] = 7
    elif dataset == 'citeseer':
        opt['hidden_dim'] = 16
        opt['input_dropout'] = 0.5
        opt['dropout'] = 0
        opt['optimizer'] = 'rmsprop'
        opt['lr'] = 0.00298
        opt['decay'] = 5e-4
        opt['self_loop_weight'] = 0.459
        opt['alpha'] = 0.936
        opt['time'] = 17.1
        opt['num_feature'] = 3703
        opt['num_class'] = 6
    elif dataset == 'pubmed':
        opt['hidden_dim'] = 16
        opt['input_dropout'] = 0.5
        opt['dropout'] = 0
        opt['optimizer'] = 'adam'
        opt['decay'] = 5e-4
        opt['lr'] = 0.00551
        opt['self_loop_weight'] = 0.752
        opt['alpha'] = 0.947
        opt['time'] = 22.0
        opt['num_feature'] = 500
        opt['num_class'] = 3
    return opt
