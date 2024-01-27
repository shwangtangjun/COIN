import os.path
import pandas as pd
import numpy as np
from sklearn import metrics
from coin import COIN, ResNet
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from load_data import load_final


def evaluate_classification_binary(y_test, y_pred, y_pred_score):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y_test, y_pred_score)

    return accuracy, auc, aupr


def calculate_weight(x_train, x_test, y_train, n_top, sigma, splits):
    filename = 'processed/distance_' + str(splits) + '.npy'

    if os.path.exists(filename):
        distance = np.load(filename)
    else:
        torch.manual_seed(0)

        model = ResNet(in_features=x_train.shape[1], hidden_dim=3, out_features=1)
        model = model.cuda()

        epochs = 300
        optimizer = optim.SGD(model.parameters(), lr=1.0, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[int(.5 * epochs), int(.75 * epochs)], gamma=0.1)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(x_train)
            loss = nn.BCELoss()(output, y_train)

            loss.backward()
            optimizer.step()

            scheduler.step()

        x = torch.cat([x_train, x_test], dim=0)
        features = model(x, return_features=True)
        features = features.detach().cpu().numpy()
        distance = np.linalg.norm(features - features[:, None], axis=-1)

        np.save(filename, distance)

    distance = torch.Tensor(distance)
    dist_n_top = torch.kthvalue(distance, n_top, dim=1, keepdim=True)[0]
    dist_sigma = torch.kthvalue(distance, sigma, dim=1, keepdim=True)[0]

    distance_truncated = distance.where(distance < dist_n_top, torch.tensor(float("inf")))
    weight = torch.exp(-(distance_truncated / dist_sigma).pow(2))

    # Symmetrically normalize the weight matrix
    d_inv_sqrt = torch.diag(weight.sum(dim=1).pow(-0.5))
    weight = d_inv_sqrt.mm(weight).mm(d_inv_sqrt)
    weight = (weight + weight.t()) / 2
    weight = weight.detach()
    return weight


def train(model, x_train, x_test, laplacian, y_train):
    x = torch.cat([x_train, x_test], dim=0)
    epochs = 300
    optimizer = optim.SGD(model.parameters(), lr=1.0, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(.5 * epochs), int(.75 * epochs)], gamma=0.1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x, laplacian)
        loss = nn.BCELoss()(output[:x_train.shape[0]], y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate(model, x_train, x_test, laplacian, y_test):
    x = torch.cat([x_train, x_test], dim=0)
    model.eval()
    output = model(x, laplacian)
    y_pred_score = output[x_train.shape[0]:]
    y_pred = torch.where(y_pred_score >= 0.5, 1., 0.)
    y_pred = y_pred.cpu().numpy()
    y_pred_score = y_pred_score.detach().cpu().numpy()
    accuracy, auc, aupr = evaluate_classification_binary(y_test, y_pred, y_pred_score)

    return accuracy, auc, aupr


def main():
    if not os.path.isdir('processed/'):  # used to store calculated weight matrix
        os.makedirs('processed/')
    resnet_accuracy_list, resnet_auc_list, resnet_aupr_list = [], [], []
    coin_accuracy_list, coin_auc_list, coin_aupr_list = [], [], []

    x, y, info, cols = load_final(data_type=['mut_important', 'cnv_del', 'cnv_amp'], cnv_levels=3)

    for i in range(10):  # splits
        print('Split:', i)
        training_set = pd.read_csv('splits/training_set_{}.csv'.format(i))
        validation_set = pd.read_csv('splits/validation_set_{}.csv'.format(i))
        testing_set = pd.read_csv('splits/test_set_{}.csv'.format(i))

        info_train = list(set(info).intersection(training_set.id))
        info_validate = list(set(info).intersection(validation_set.id))
        info_test = list(set(info).intersection(testing_set.id))

        ind_train = info.isin(info_train)
        ind_validate = info.isin(info_validate)
        ind_test = info.isin(info_test)

        x_train = x[ind_train]
        x_test = x[ind_test]
        x_validate = x[ind_validate]

        y_train = y[ind_train]
        y_test = y[ind_test]
        y_validate = y[ind_validate]

        x_train = torch.Tensor(x_train).cuda()
        y_train = torch.Tensor(y_train).cuda()
        x_test = torch.Tensor(x_test).cuda()

        adj = calculate_weight(x_train, x_test, y_train, 40, 20, splits=i).cuda()
        diagonal = torch.diag(torch.sum(adj, dim=1))
        laplacian = diagonal - adj

        for init_seed in range(10):  # 10 random inits
            # ResNet
            torch.manual_seed(init_seed)
            model = COIN(in_features=x_train.shape[1], hidden_dim=3, out_features=1, layer_num=0,
                         sigma2=0.)
            model = model.cuda()

            train(model, x_train, x_test, laplacian, y_train)

            accuracy, auc, aupr = evaluate(model, x_train, x_test, laplacian, y_test)

            resnet_accuracy_list.append(accuracy)
            resnet_auc_list.append(auc)
            resnet_aupr_list.append(aupr)

            # COIN
            torch.manual_seed(init_seed)
            model = COIN(in_features=x_train.shape[1], hidden_dim=3, out_features=1, layer_num=40,
                         sigma2=0.2)
            model = model.cuda()

            train(model, x_train, x_test, laplacian, y_train)

            accuracy, auc, aupr = evaluate(model, x_train, x_test, laplacian, y_test)

            coin_accuracy_list.append(accuracy)
            coin_auc_list.append(auc)
            coin_aupr_list.append(aupr)

    resnet_accuracy_list = np.array(resnet_accuracy_list)
    resnet_auc_list = np.array(resnet_auc_list)
    resnet_aupr_list = np.array(resnet_aupr_list)

    coin_accuracy_list = np.array(coin_accuracy_list)
    coin_auc_list = np.array(coin_auc_list)
    coin_aupr_list = np.array(coin_aupr_list)

    print('ResNet')
    print('Mean accuracy:', np.mean(resnet_accuracy_list))
    print('Mean auc:', np.mean(resnet_auc_list))
    print('Mean aupr:', np.mean(resnet_aupr_list))

    print('COIN')
    print('Mean accuracy:', np.mean(coin_accuracy_list))
    print('Mean auc:', np.mean(coin_auc_list))
    print('Mean aupr:', np.mean(coin_aupr_list))


if __name__ == '__main__':
    main()
