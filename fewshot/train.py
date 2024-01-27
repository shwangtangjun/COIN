import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from utils import get_tqdm, get_configuration, get_dataloader, get_embedded_feature, get_base_mean
from utils import compute_confidence_interval, calculate_weight
from coin import COIN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int, help='seed for training')
parser.add_argument("--dataset", choices=['mini', 'tiered', 'cub'], type=str, default='mini')
parser.add_argument("--backbone", choices=['resnet18', 'wideres'], type=str, default='resnet18')
parser.add_argument("--query_per_class", default=15, type=int,
                    help="number of unlabeled query sample per class")
parser.add_argument("--way", default=5, type=int, help="5-way-k-shot")
parser.add_argument("--test_iter", default=10000, type=int,
                    help="test on 10000 tasks and output average accuracy")
parser.add_argument("--shot", choices=[1, 5], type=int, default=1)
parser.add_argument('--silent', action='store_true', help='call --silent to disable tqdm')

parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument("--sigma2", type=float, help='strength of each diffusion layer', default=0.5)
parser.add_argument("--layer_num", type=int, help='number of diffusion layers, 0 means no diffusion',
                    default=10)
parser.add_argument("--n_top", type=int, default=8)
parser.add_argument("--sigma", type=int, default=4)

args = parser.parse_args()


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    data_path, split_path, save_path, num_classes = get_configuration(args.dataset, args.backbone)

    # On novel class: get the output of embedding function (backbone)
    # On base class: get the output average of embedding function (backbone), used for centering
    train_loader = get_dataloader(data_path, split_path, 'train')
    test_loader = get_dataloader(data_path, split_path, 'test')
    embedded_feature = get_embedded_feature(test_loader, save_path, args.silent)
    base_mean = get_base_mean(train_loader, save_path, args.silent)

    acc_list = []
    tqdm_test_iter = get_tqdm(range(args.test_iter), args.silent)

    for _ in tqdm_test_iter:
        acc = single_trial(embedded_feature, base_mean)
        acc_list.append(acc)

        if not args.silent:
            tqdm_test_iter.set_description(
                'Test on few-shot tasks. Accuracy:{:.2f}'.format(np.mean(acc_list)))

    acc_mean, acc_conf = compute_confidence_interval(acc_list)
    print('Accuracy:{:.2f}'.format(acc_mean))
    print('Conf:{:.2f}'.format(acc_conf))


def sample_task(embedded_feature):
    """
    Sample a single few-shot task from novel classes
    """
    sample_class = random.sample(list(embedded_feature.keys()), args.way)
    train_data, test_data, test_label, train_label = [], [], [], []

    for i, each_class in enumerate(sample_class):
        samples = random.sample(embedded_feature[each_class], args.shot + args.query_per_class)

        train_label += [i] * args.shot
        test_label += [i] * args.query_per_class
        train_data += samples[:args.shot]
        test_data += samples[args.shot:]

    return np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label)


def single_trial(embedded_feature, base_mean):
    train_data, test_data, train_label, test_label = sample_task(embedded_feature)

    train_data, test_data, train_label, test_label, base_mean = torch.tensor(train_data), torch.tensor(
        test_data), torch.tensor(train_label), torch.tensor(test_label), torch.tensor(base_mean)

    # Centering and Normalization
    train_data = train_data - base_mean
    train_data = train_data / torch.norm(train_data, dim=1, keepdim=True)
    test_data = test_data - base_mean
    test_data = test_data / torch.norm(test_data, dim=1, keepdim=True)

    # Cross-Domain Shift
    eta = train_data.mean(dim=0, keepdim=True) - test_data.mean(dim=0, keepdim=True)
    test_data = test_data + eta

    inputs = torch.cat([train_data, test_data], dim=0)
    weight = calculate_weight(inputs)
    inputs, train_label, weight = inputs.cuda(), train_label.cuda(), weight.cuda()
    model = COIN(in_features=inputs.shape[1], sigma2=args.sigma2, layer_num=args.layer_num).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)], gamma=0.1)

    diagonal = torch.diag(weight.sum(dim=1))
    laplacian = diagonal - weight

    for epoch in range(args.epochs):
        train(model, inputs, laplacian, train_label, optimizer)
        scheduler.step()

    outputs = model(inputs, laplacian)

    # get the accuracy only on query data
    pred = outputs.argmax(dim=1)[args.way * args.shot:].cpu()
    acc = torch.eq(pred, test_label).float().mean().cpu().numpy() * 100
    return acc


def train(model, inputs, laplacian, train_label, optimizer):
    outputs = model(inputs, laplacian)
    outputs = torch.log(outputs)
    loss = nn.NLLLoss()(outputs[:args.way * args.shot], train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    main()
