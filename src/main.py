import random
import numpy as np
import logging
import sys
import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import warnings
from pathlib import Path
from torch_geometric.utils.sparse import dense_to_sparse
from causal import Learn_structure
from SetParser import arg_parse
from utils import get_data_loaders
from gutils import compute_ppr

from model import MVGRL, local_global_loss_, ClsHeader
from sklearn.model_selection import GridSearchCV, StratifiedKFold

os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
warnings.filterwarnings("ignore")


def process_data(data):
    num_node = len(data.batch)
    if data.edge_attr == None:
        data.edge_attr = torch.ones(data.edge_index.shape[1])
    adj = torch.sparse_coo_tensor(data.edge_index,
                                  data.edge_attr.squeeze(),   #  ,
                                  (num_node, num_node)).to_dense()
    data_diff = data.clone().detach()
    data_diff.edge_index, data_diff.edge_attr = dense_to_sparse(compute_ppr(adj))
    return data, data_diff


def train(model, loader, args, ft=False):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cnt_wait = 0
    best = 1e9
    if ft:
        nb_epochs = args.ft_epoch
    else:
        nb_epochs = args.epoch
    start_time = time.time()
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        itr = 0
        for data in loader:
            data, data_diff = process_data(data)
            data_diff = data_diff.to(device)
            data = data.to(device)
            model.train()
            optimiser.zero_grad()
            lv1, gv1, lv2, gv2, pred = model(data, data_diff)
            loss1 = local_global_loss_(lv1, gv2, data.ptr, 'JSD')
            loss2 = local_global_loss_(lv2, gv1, data.ptr, 'JSD')
            loss3 = CELoss(pred, data.y)
            loss = loss1 + loss2 + loss3
            epoch_loss += loss
            loss.backward()
            optimiser.step()
            itr += 1

        epoch_loss /= itr
        print('Epoch: {0}, Loss: {1:0.6f}'.format(epoch, epoch_loss))

        if epoch_loss < best:
            best = epoch_loss
            cnt_wait = 0
            torch.save(model.state_dict(), f'{args.dataset}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            end_time = time.time()
            print(f"time: {end_time-start_time}")
            break

    end_time = time.time()
    print(f"Training time: {end_time - start_time}")


def Eval(model, loader):
    print('evaluation')
    # model.load_state_dict(torch.load(args.dataset+'.pkl'))

    accuracies = 0.
    n_test = 0
    for data in loader['test']:
        n_test += len(data.y)
        data, data_diff = process_data(data)
        data_diff = data_diff.to(device)
        data = data.to(device)
        _, _, _, _, pred = model(data, data_diff)
        accuracies += torch.sum(pred.argmax(-1).view(-1) == data.y.view(-1))
        print(n_test)
    accuracies /= n_test

    '''
    model.eval()
    embeds = []
    labels = []
    for data in loader['train']:
        data, data_diff = process_data(data)
        data = data.to(device)
        data_diff = data_diff.to(device)
        embeds.append(model.embed(data, data_diff))
        labels.append(data.y)

    x_train = torch.cat(embeds, dim=0).to(device) # .to(device)  # .cpu().numpy()
    y_train = torch.cat(labels).squeeze().float().to(device)  # .to(device) # .cpu().numpy()

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
    classifier.fit(x_train, y_train)
    accuracies = accuracy_score(y_train, classifier.predict(x_train))
    print(f"acc {accuracies}")
    
    cls = ClsHeader(args.hid_units, num_class)
    cls.cuda()
    optim = Adam(cls.parameters(), lr=lr)
    cls.train()
    for _ in range(args.epoch):
        optim.zero_grad()
        pred = cls(x_train)
        loss = F.binary_cross_entropy(pred.max(dim=1)[0], y_train)
        loss.backward()
        optim.step()
    cls.eval()
    # pred = cls(x_test)
    '''
    print(f"acc {accuracies}")

    return accuracies


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


print('Hello@')
if __name__ == '__main__':
    # set parameters
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    args = arg_parse()
    data_dir = args.dataset_dir
    data_dir = Path('../' + data_dir)
    dataset_name = args.dataset

    random_state = args.seed
    set_seed(random_state)
    nb_epochs = args.epoch
    batch_size = args.batch_size
    hid_units = args.hid_units  # hid_units = 512
    num_layer = args.num_layer
    lr = args.lr
    patience = args.patience
    l2_coef = 0.0
    splits = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

    # Logger file
    os.makedirs("result", exist_ok=True)
    log_file = "log_{}.txt".format(args.dataset)
    log_file = os.path.join("result", log_file)
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    logger.info("Dataset: {}".format(args.dataset))

    # preprocess and pretrain
    loader, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
        data_dir, dataset_name, batch_size, splits, random_state)

    ft_size = x_dim

    model = MVGRL(ft_size, hid_units, num_class, num_layer).to(device)
    CELoss = nn.CrossEntropyLoss(reduction="mean")
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    train(model, loader['train'], args)
    acc = Eval(model, loader)
    logger.info(f"acc: {acc}")

    # Causal structure initialization
    '''
    Causal_structure = adj.clone().detach()
    # Causal structure learning
    for itr in range(args.itr):
        Causal_structure = Learn_structure(model, data, Causal_structure, args)
        data_aug = [feat, Causal_structure, diff, labels, num_nodes]
        train(model, data_aug, ft=True, args=args)
        acc_mean, acc_std = Eval(model, data)
        train(model, data, ft=True, args=args)
        logger.info(f"acc mean: {acc_mean}, acc std: {acc_std}")
        torch.save(Causal_structure, 'Causal_structure.pt')
    '''

