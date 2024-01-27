import numpy as np
import scipy.sparse as sp
import torch
import sys
sys.path.append("../..")
from data_process.make_dataset import get_dataset, get_train_val_test_split
from scipy.linalg import expm
from torch_geometric.data import Data


def load_data(dataset_str, random_state):
    data_path = "../../data/" + dataset_str + ".npz"

    adj, features, labels = get_dataset(dataset_str, data_path, standardize=True, train_examples_per_class=20,
                                        val_examples_per_class=30)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=20,
                                                            val_examples_per_class=30, test_size=None)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to_dense().numpy()
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
        opt['hidden_layers'] = 1
        opt['hidden_units'] = 64
        opt['dropout'] = 0.5
        opt['lr'] = 0.01
        opt['weight_decay'] = 0.0861202357056583
        opt['t'] = 5.0
        opt['eps'] = 0.0001
    elif dataset == 'citeseer':
        opt['hidden_layers'] = 1
        opt['hidden_units'] = 64
        opt['dropout'] = 0.5
        opt['lr'] = 0.01
        opt['weight_decay'] = 10.0
        opt['t'] = 4.0
        opt['eps'] = 0.0009
    elif dataset == 'pubmed':
        opt['hidden_layers'] = 1
        opt['hidden_units'] = 64
        opt['dropout'] = 0.5
        opt['lr'] = 0.01
        opt['weight_decay'] = 0.04
        opt['t'] = 3.0
        opt['eps'] = 0.0001
    return opt


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_heat_matrix(adj_matrix: np.ndarray, t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm


class torch_geometric_like_dataset():
    def __init__(self, rewired_edge_index, rewired_edge_attr, features, labels, idx_train, idx_val, idx_test):
        self.num_classes = labels.max().item() + 1
        self.num_nodes = features.shape[0]
        self.num_features = features.shape[1]
        self.data = Data(x=features, y=labels, edge_index=rewired_edge_index, edge_attr=rewired_edge_attr)

        self.data.train_mask = self.create_mask(idx_train)
        self.data.val_mask = self.create_mask(idx_val)
        self.data.test_mask = self.create_mask(idx_test)

    def create_mask(self, idx):
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask


def rewired_matrix_to_edge_form(rewired_matrix):
    edges_i = []
    edges_j = []
    edge_attr = []
    for i, row in enumerate(rewired_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(rewired_matrix[i, j])
    edge_index = [edges_i, edges_j]
    return torch.LongTensor(edge_index), torch.FloatTensor(edge_attr)
