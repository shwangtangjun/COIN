import argparse
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_splits', type=int, default=10, help='Number of different missing choice.')
parser.add_argument('--num_inits', type=int, default=10, help='Number of different initializations.')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--layer_num', type=int, default=10)
parser.add_argument('--sigma2', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--method', type=str, default='coin', help='See code below for choice')
parser.add_argument('--missing_rate', type=float, default=0.9)

parser.add_argument('--verbose', action='store_true')
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # specify which GPU(s) to be used

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from coin import COIN, GCN, MLP, DCRNNNet, MPNNNet, GConvGRUNet, A3TGCNNet, EvolveGCNNet
import time
from load_data import EnglandCovidDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.utils import to_dense_adj
from sklearn.linear_model import LinearRegression


def normalize_adj(adj, normalization="symmetric"):
    # only used for our method
    adj.fill_diagonal_(0.)
    if normalization == "symmetric":
        rowsum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        mx = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == "row":
        rowsum = torch.sum(adj, dim=1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, adj)
    else:
        raise NotImplementedError
    return mx


def train(model, optimizer, train_dataset):
    model.train()
    optimizer.zero_grad()

    loss = torch.tensor(0.).cuda()
    for time, snapshot in enumerate(train_dataset):
        valid_idx = ~torch.isnan(snapshot.y)

        # COIN
        if args.method == 'coin':
            features = snapshot.x.cuda()
            labels = snapshot.y.cuda()
            adj = to_dense_adj(edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr).cuda().squeeze()
            adj = normalize_adj(adj, normalization='symmetric')
            diagonal = torch.diag(torch.sum(adj, dim=1))
            laplacian = diagonal - adj
            output = model(features, laplacian).squeeze()
            loss += nn.MSELoss()(output[valid_idx], labels[valid_idx])
        # GCN and others
        else:
            output = model(snapshot.x.cuda(), snapshot.edge_index.cuda(), snapshot.edge_attr.cuda()).squeeze()
            loss += nn.MSELoss()(output[valid_idx], snapshot.y.cuda()[valid_idx])

    loss /= time + 1

    loss.backward()
    optimizer.step()

    if args.verbose:
        print('Train loss: {:.4f}'.format(loss.detach().cpu().numpy()), end=4 * ' ')


@torch.no_grad()
def val(model, val_dataset):
    model.eval()

    loss = torch.tensor(0.).cuda()
    for time, snapshot in enumerate(val_dataset):
        valid_idx = ~torch.isnan(snapshot.y)

        # COIN
        if args.method == 'coin':
            features = snapshot.x.cuda()
            labels = snapshot.y.cuda()
            adj = to_dense_adj(edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr).cuda().squeeze()
            adj = normalize_adj(adj, normalization='symmetric')
            diagonal = torch.diag(torch.sum(adj, dim=1))
            laplacian = diagonal - adj
            output = model(features, laplacian).squeeze()
            loss += nn.MSELoss()(output[valid_idx], labels[valid_idx])
        # GCN and others
        else:
            output = model(snapshot.x.cuda(), snapshot.edge_index.cuda(), snapshot.edge_attr.cuda()).squeeze()
            loss += nn.MSELoss()(output[valid_idx], snapshot.y.cuda()[valid_idx])
    loss /= time + 1

    if args.verbose:
        print('Val loss: {:.4f}'.format(loss.detach().cpu().numpy()), end=4 * ' ')
    return loss


@torch.no_grad()
def test(model, test_dataset):
    model.eval()

    loss = torch.tensor(0.).cuda()
    for time, snapshot in enumerate(test_dataset):
        # COIN
        if args.method == 'coin':
            features = snapshot.x.cuda()
            labels = snapshot.y.cuda()
            adj = to_dense_adj(edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr).cuda().squeeze()
            adj = normalize_adj(adj, normalization='symmetric')
            diagonal = torch.diag(torch.sum(adj, dim=1))
            laplacian = diagonal - adj
            output = model(features, laplacian).squeeze()
            loss += nn.MSELoss()(output, labels)
        # GCN and others
        else:
            output = model(snapshot.x.cuda(), snapshot.edge_index.cuda(), snapshot.edge_attr.cuda()).squeeze()
            loss += nn.MSELoss()(output, snapshot.y.cuda())
    loss /= time + 1

    return loss


def run_single_trial_of_single_split(train_dataset, val_dataset, test_dataset, torch_seeds):
    torch.manual_seed(torch_seeds)
    torch.cuda.manual_seed(torch_seeds)

    in_channels = next(iter(train_dataset)).x.shape[1]
    if args.method == 'mlp':
        model = MLP(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=1,
                    dropout=args.dropout)
    elif args.method == 'gcn':
        model = GCN(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=1,
                    dropout=args.dropout)
    elif args.method == 'coin':
        model = COIN(in_features=in_channels, hidden_dim=args.hidden_dim, out_features=1,
                     layer_num=args.layer_num, sigma2=args.sigma2, dropout=args.dropout)
    elif args.method == 'dcrnn':
        model = DCRNNNet(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=1,
                         dropout=args.dropout)
    elif args.method == 'mpnn':
        model = MPNNNet(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=1,
                        dropout=args.dropout)
    elif args.method == 'gconvgru':
        model = GConvGRUNet(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=1,
                            dropout=args.dropout)
    elif args.method == 'a3tgcn':
        model = A3TGCNNet(in_channels=in_channels, hidden_channels=args.hidden_dim, out_channels=1,
                          dropout=args.dropout)
    elif args.method == 'evolvegcn':
        model = EvolveGCNNet(in_channels=in_channels, out_channels=1, dropout=args.dropout)
    elif args.method == 'lr':
        x_train = np.row_stack(train_dataset.features)
        y_train = np.hstack(train_dataset.targets)
        x_test = np.row_stack(test_dataset.features)
        y_test = np.hstack(test_dataset.targets)

        valid_idx = ~np.isnan(y_train)
        model = LinearRegression().fit(x_train[valid_idx], y_train[valid_idx])
        pred = model.predict(x_test)
        loss = nn.MSELoss()(torch.tensor(y_test), torch.tensor(pred))
        return loss
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.cuda()

    for epoch in range(args.epochs):
        train(model, optimizer, train_dataset)
        if args.verbose:
            test_loss = test(model, test_dataset)
            print('Test loss: {:.4f}'.format(test_loss.detach().cpu().numpy()))

    loss = test(model, test_dataset)
    loss = loss.cpu().numpy()

    return loss


def run_single_split(seed):
    random_state = np.random.RandomState(seed)

    loader = EnglandCovidDatasetLoader()
    dataset = loader.get_dataset(missing_rate=args.missing_rate, missing_random_state=random_state)
    train_dataset, val_test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
    val_dataset, test_dataset = temporal_signal_split(val_test_dataset, train_ratio=0.2 / 0.8)

    torch_seeds = random_state.randint(0, 1000000, args.num_inits)  # 20 trials for each split
    loss_list = []
    for i in range(args.num_inits):
        loss = run_single_trial_of_single_split(train_dataset, val_dataset, test_dataset, torch_seeds[i])
        loss_list.append(loss)
    return np.array(loss_list)


def main():
    time1 = time.time()
    random_state = np.random.RandomState(args.seed)
    single_split_seed = random_state.randint(0, 1000000, args.num_splits)  # 100 random splits

    total_loss_list = []
    for i in range(args.num_splits):
        loss_of_single_split = run_single_split(single_split_seed[i])
        print(loss_of_single_split)
        total_loss_list.append(loss_of_single_split)

    time2 = time.time()
    print(np.mean(total_loss_list))
    print(np.std(total_loss_list))

    print('layer_num=', args.layer_num)
    print('sigma2=', args.sigma2)
    print('time=', time2 - time1)


if __name__ == '__main__':
    main()
