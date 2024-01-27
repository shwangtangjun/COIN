# COIN
This repository contains the code for COIN (COnvection dIffusion Network) implemented with PyTorch. 

## Introduction
Inspired by scale-space theory and the connection between ODE, PDE and ResNets, we provide a theoretically certified framework for describing neural networks, which not only covers various existing network structure, but also illuminates new thinking for designing networks.

## Graph Node Classification
### 1. COIN
Users can test our COIN on dataset Cora, Citeseer and Pubmed for 100 random dataset splits and 20 random initializations each. One should provide ```layer_num``` and ```sigma2```. Specific parameter choice for reproducing results in paper is provided in the Materials and Methods.
```
python train.py --dataset cora --layer_num 20 --sigma2 0.35
```
One may use small ```num_splits``` and ```num_inits``` for quick test. 
```
python train.py --dataset cora --layer_num 20 --sigma2 0.35 --num_splits 1 --num_inits 1
```
On a single NVIDIA GeForce RTX 3090, the line above should take less than 1 second on average to converge.
Python dependencies include PyTorch, NumPy and Scipy.

### 2. Re-implement other papers
We also provide the code to reproduce other papers in [graph/reproduce](./graph/reproduce). To run, just go to target directory and run the corresponding ```train_**.py``` file,
```
cd appnp/
python train_appnp.py --dataset cora
```
The best training parameters, collected from papers, have been stored in the ```utils_**.py```. One may also use small ```num_splits``` and ```num_inits``` for quick test. 
The environment requirement varies with papers. For most papers, PyTorch and [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) should be sufficient. For methods that use ODE, [torchdiffeq](https://github.com/rtqichen/torchdiffeq) should be installed. Deep Graph Library([DGL](https://www.dgl.ai/)) is required for GCDE. [torch-sparse and torch-scatter](https://github.com/rusty1s/pytorch_sparse) are required for Difformer. For GRAND, the dependencies are complex, and we recommend users to go to the original [repo](https://github.com/twitter-research/graph-neural-pde) for reference.

## Few-shot learning
### 1. Dataset
Download [miniImageNet](https://mega.nz/file/2ldRWQ7Y#U_zhHOf0mxoZ_WQNdvv4mt1Ke3Ay9YPNmHl5TnOVuAU), [tieredImageNet](https://mega.nz/file/r1kmyAgR#uMx7x38RScStpTZARKL2DwTfkD1eVIgbilL4s20vLhI) and [CUB-100](https://mega.nz/file/axUDACZb#ve0NQdmdj_RhhQttONaZ8Tgaxdh4A__PASs_OCI6cSk). Unpack these dataset in to corresponding dataset name directory in [data/](./fewshot/data/).

### 2. Backbone Training
You can download pretrained models on base classes [here](https://mega.nz/file/f5lDUJSY#E6zdNonvpPP5nq7cx_heYgLSU6vxCrsbvy4SNr88MT4), and unpack pretrained models in fewshot/saved_models/.

Or you can train from scratch by running [train_backbone.py](./fewshot/backbone/train_backbone.py).

```
python train_backbone.py --dataset mini --backbone resnet18 --silent --epochs 100
```
Note that backbone training requires hours.

### 3. Few-shot Classification
Run [train.py](./fewshot/train.py) with specified arguments for few-shot classification.  Specific parameter choice for reproducing results in paper is provided in the the Materials and Methods. See argument description for help.
```
python train.py --dataset mini --backbone resnet18 --shot 1 --layer_num 10 --sigma2 0.5
```
On a single NVIDIA GeForce RTX 3090, the line above should take around 30 mins, for 10000 evaluation tasks.

## COVID-19 case prediction
Users can test our COIN on COVID-19 case prediction task with missing data, for 10 random masking and 10 random initializations each.
```
python train.py --layer_num 10 --sigma2 0.5
```
Python dependencies include PyTorch, [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/),[Pytorch-Geometric-Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html), [torch-sparse and torch-scatter](https://github.com/rusty1s/pytorch_sparse).

## Prostrate cancer classification
### 1. Dataset
The dataset is collected from the supplementary information of [The long tail of oncogenic drivers in prostate cancer](https://www.nature.com/articles/s41588-018-0078-z#Sec17) and [Biologically informed deep neural network for prostate cancer classification and discovery](https://www.nature.com/articles/s41586-021-03922-4). To run our program, users need to download three files, [Supplementary Table 2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM4_ESM.txt), [Supplementary Table 3](https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM5_ESM.xlsx) and [Supplementary Table 10](https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt) from supplementary information of the first paper. The other two necessary data files have already been provided in the repo.

### 2. Train
```
python train.py
```
The program should require some time and large cpu memory at first time, due to the exact computation of pairwise distance. We will try to optimize it using approximate algorithm in the future. After the distance is calculated, it will be stored, and further evaluation will be much faster.

Python dependencies include PyTorch, Pandas, and openpyxl. The latter two are only used for data input.
