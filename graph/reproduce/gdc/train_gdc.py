import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils_gdc import load_data, accuracy, get_best_params, get_heat_matrix, get_clipped_matrix, \
    torch_geometric_like_dataset, rewired_matrix_to_edge_form
from gcn import GCN
import copy
import time

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_splits', type=int, default=100, help='Number of different splits.')
parser.add_argument('--num_inits', type=int, default=20, help='Number of different initializations.')

parser.add_argument('--max_epochs', type=int, default=10000, help='Max uumber of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience.')

parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()

device = torch.device('cuda:' + args.device)


def train(model, optimizer, features, labels, idx_train, dataset):
    model.train()
    optimizer.zero_grad()

    output = model(dataset.data)
    loss = nn.CrossEntropyLoss()(output[idx_train], labels[idx_train])

    loss.backward()
    optimizer.step()

    if args.verbose:
        acc_train = accuracy(output[idx_train], labels[idx_train])
        print('Train loss: {:.4f}'.format(loss.detach().cpu().numpy()), end=4 * ' ')
        print('Train acc: {:.2f}%'.format(acc_train.cpu().numpy() * 100), end=2 * ' ')


@torch.no_grad()
def val(model, features, labels, idx_val, dataset):
    model.eval()

    output = model(dataset.data)

    loss = nn.CrossEntropyLoss()(output[idx_val], labels[idx_val])
    acc = accuracy(output[idx_val], labels[idx_val])
    if args.verbose:
        print('Val loss: {:.4f}'.format(loss.detach().cpu().numpy()), end=4 * ' ')
        print('Val Acc: {:.2f}%'.format(acc.cpu().numpy() * 100), end=2 * ' ')
    return loss, acc


@torch.no_grad()
def test(model, features, labels, idx_test, dataset):
    model.eval()

    output = model(dataset.data)
    acc_test = accuracy(output[idx_test], labels[idx_test])

    return acc_test


def run_single_trial_of_single_split(rewired_edge_index, rewired_edge_attr, features, labels, idx_train, idx_val,
                                     idx_test, torch_seeds, opt):
    torch.manual_seed(torch_seeds)
    torch.cuda.manual_seed(torch_seeds)

    rewired_edge_index = rewired_edge_index.to(device)
    rewired_edge_attr = rewired_edge_attr.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    dataset = torch_geometric_like_dataset(rewired_edge_index, rewired_edge_attr, features, labels, idx_train, idx_val,
                                           idx_test)

    model = GCN(dataset, hidden=opt['hidden_layers'] * [opt['hidden_units']], dropout=opt['dropout'])
    model = model.to(device)

    optimizer = optim.Adam([{'params': model.non_reg_params, 'weight_decay': 0},
                            {'params': model.reg_params, 'weight_decay': opt['weight_decay']}], lr=opt['lr'])

    val_loss_min = np.inf
    val_acc_max = 0
    patience_step = 0
    best_state_dict = None

    val_loss_list = []
    val_acc_list = []

    for epoch in range(args.max_epochs):
        if args.verbose:
            print('Epoch: {}'.format(epoch + 1), end=4 * ' ')
        train(model, optimizer, features, labels, idx_train, dataset)

        val_loss, val_acc = val(model, features, labels, idx_val, dataset)

        val_loss = val_loss.detach().cpu().numpy()
        val_acc = val_acc.detach().cpu().numpy()
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if args.verbose:
            test_acc = test(model, features, labels, idx_test, dataset)
            print('Test acc: {:.2f}%'.format(test_acc.cpu().numpy() * 100))

        if val_loss < val_loss_min or val_acc > val_acc_max:
            val_loss_min = np.min((val_loss, val_loss_min))
            val_acc_max = np.max((val_acc, val_acc_max))
            patience_step = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_step += 1

        if patience_step >= args.patience:
            model.load_state_dict(best_state_dict)
            break

    model.load_state_dict(best_state_dict)
    acc = test(model, features, labels, idx_test, dataset)
    acc = acc.cpu().numpy()

    return acc


def run_single_split(seed):
    random_state = np.random.RandomState(seed)

    opt = get_best_params(args.dataset)

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, random_state)

    # for each dataset, we only need to calculate the rewired matrix once
    if not os.path.isdir('rewired_adj_matrix/'):
        os.makedirs('rewired_adj_matrix/')
    if not os.path.isfile('rewired_adj_matrix/' + args.dataset + '_heat_edge_index.pt'):
        rewired_matrix = get_heat_matrix(adj, t=opt['t'])
        rewired_matrix = get_clipped_matrix(rewired_matrix, eps=opt['eps'])
        rewired_edge_index, rewired_edge_attr = rewired_matrix_to_edge_form(rewired_matrix)
        torch.save(rewired_edge_index, 'rewired_adj_matrix/' + args.dataset + '_heat_edge_index.pt')
        torch.save(rewired_edge_attr, 'rewired_adj_matrix/' + args.dataset + '_heat_edge_attr.pt')
    else:
        rewired_edge_index = torch.load('rewired_adj_matrix/' + args.dataset + '_heat_edge_index.pt')
        rewired_edge_attr = torch.load('rewired_adj_matrix/' + args.dataset + '_heat_edge_attr.pt')

    torch_seeds = random_state.randint(0, 1000000, args.num_inits)  # 20 trials for each split
    acc_list = []
    for i in range(args.num_inits):
        acc = run_single_trial_of_single_split(rewired_edge_index, rewired_edge_attr, features, labels, idx_train,
                                               idx_val, idx_test, torch_seeds[i], opt)
        acc_list.append(acc)
    return np.array(acc_list)


def main():
    time1 = time.time()
    random_state = np.random.RandomState(args.seed)
    single_split_seed = random_state.randint(0, 1000000, args.num_splits)  # 100 random splits

    total_acc_list = []
    for i in range(args.num_splits):
        acc_of_single_split = run_single_split(single_split_seed[i])
        print(acc_of_single_split)
        total_acc_list.append(acc_of_single_split)

    time2 = time.time()
    print(np.mean(total_acc_list) * 100)
    print(np.std(total_acc_list) * 100)

    print('time=', time2 - time1)


if __name__ == '__main__':
    main()
