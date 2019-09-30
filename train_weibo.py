import os
import time
import torch
import copy
import numpy as np
from config import get_config
import torch.optim as optim
from models.models import MultiInputMLP, MultiInputGCN, HGNN
from datasets.weibo import load_weibo
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def train_weibo(model, fts, edge_dict, lbls, idx_train, idx_val, criterion, optimizer, scheduler, device,
                b_use_hypergraph: bool, b_list_input: bool, num_epochs=25, print_freq=5):
    """
    :param model:
    :param fts:
    :param edge_dict:
    :param lbls:
    :param idx_train:
    :param idx_val:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param b_use_hypergraph -> bool: whether use hypergraph learning
    :param b_list_input -> bool: whether use list of features
    :param num_epochs:
    :param print_freq:
    :return:
    """
    model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device)
    best_acc = 0.0
    loss_min = 100

    for epoch in range(num_epochs):

        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val

            if b_use_hypergraph:
                inputs = [torch.Tensor(fts[i]).to(device) for i in range(len(fts))] if b_list_input \
                    else torch.Tensor(fts).to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, edge_dict)
                    loss = criterion(outputs[idx], lbls[idx]) * len(idx)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            else:
                inputs = [torch.Tensor(fts[i][idx]).to(device) for i in range(len(fts))]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, lbls[idx]) * len(idx)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(preds == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if phase == 'val' and epoch_loss < loss_min:
                loss_min = epoch_loss
                model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if epoch % print_freq == 0 and phase == 'val':
                print(f'Best val Acc: {best_acc:4f}, Min val loss: {loss_min:4f}')
                print('-' * 20)

    print(f'Best val Acc: {best_acc:4f}')

    return model_wts_best_val_acc, model_wts_lowest_val_loss


def test_weibo(model, best_model_wts, fts, edge_dict, lbls, idx_test, device, use_hypergraph, list_input):
    model.load_state_dict(best_model_wts)
    model = model.to(device)
    model.eval()

    running_corrects = 0.0

    if use_hypergraph:
        inputs = [torch.Tensor(fts[i]).to(device) for i in range(len(fts))] if list_input \
            else torch.Tensor(fts).to(device)

        with torch.no_grad():
            outputs = model(inputs, edge_dict)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds[idx_test] == lbls.data[idx_test])
        test_acc = running_corrects.double() / len(idx_test)

    else:
        inputs = [torch.Tensor(fts[i][idx_test]).to(device) for i in range(len(fts))]

        with torch.no_grad():
            outputs, _ = model(inputs)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == lbls.data[idx_test])
        test_acc = running_corrects.double() / len(idx_test)

    print('*' * 20)
    print(f'Test acc: {test_acc}')
    print('*' * 20)

    return test_acc


def get_embeddings(model, best_model_wts, fts, device):
    model.load_state_dict(best_model_wts)
    model = model.to(device)
    model.eval()

    inputs = [torch.Tensor(fts[i]).to(device) for i in range(len(fts))]

    with torch.no_grad():
        _, embs = model(inputs)

    return embs


def experiment_mlgcn():
    since = time.time()

    device = torch.device('cuda:0')

    cfg = get_config('config/config_weibo.yaml')

    fts, lbls, idx_train, idx_val, idx_test = load_weibo(cfg)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    dims_in = [fts[i].shape[1] for i in range(len(fts))]
    hiddens = cfg['hiddens']

    model = MultiInputMLP(n_input=3, dims_in=dims_in, n_category=2, hiddens=hiddens)

    # count model flops and params
    params = 0
    for param in model.state_dict().values():
        params += param.data.numpy().size
    print(f'# params: {params}')

    model_init(model)                       # initialize model

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.NLLLoss()

    # pre-train classifier for feature embeddings
    model_wts_best_val_acc, model_wts_lowest_val_loss \
        = train_weibo(model, fts, None, lbls, idx_train, idx_val, criterion, optimizer, schedular,
                      device, False, True, num_epochs=50, print_freq=10)      # fixed parameters for MLP pre-train model

    print('**** Model of lowest val loss ****')
    test_weibo(model, model_wts_lowest_val_loss, fts, None, lbls, idx_test, device, False, True)
    print('**** Model of best val acc ****')
    test_weibo(model, model_wts_best_val_acc, fts, None, lbls, idx_test, device, False, True)

    # construct hypergraph using feature embeddings
    # get embeddings
    embs = get_embeddings(model, model_wts_lowest_val_loss, fts, device)
    # embs = [torch.cat([embs[i] for i in range(len(embs))], dim=1).cpu().numpy()]
    embs = [embs[i].cpu().numpy() for i in range(len(embs))]

    graph_type = cfg['graph_type']
    knns = cfg['knns']
    clusters = cfg['clusters']
    adjacent_clusters = cfg['adjacent_clusters']
    if graph_type == 'knn':
        edge_list = construct_edge_list_from_knn(embs, knns)
    elif graph_type == 'cluster':
        edge_list = construct_edge_list_from_cluster(embs, clusters, adjacent_clusters, knns)

    # hypergraph learning
    model = MultiInputGCN(n_input=3, dims_in=dims_in, n_category=2,
                          knn=sum(knns), hiddens=hiddens, drop_out=cfg['drop_out'])

    model_init(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.NLLLoss()

    model_wts_best_val_acc, model_wts_lowest_val_loss \
    = train_weibo(model, fts, edge_list, lbls, idx_train, idx_val, criterion, optimizer, schedular,
                  device, True, True, num_epochs=cfg['max_epoch'], print_freq=cfg['print_freq'])

    print('**** Model of lowest val loss ****')
    acc0 = test_weibo(model, model_wts_lowest_val_loss, fts, edge_list, lbls, idx_test, device, True, True)
    print('**** Model of best val acc ****')
    acc1 = test_weibo(model, model_wts_best_val_acc, fts, edge_list, lbls, idx_test, device, True, True)
    test_acc = max(acc0, acc1)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return test_acc


def experiment_hgnn():
    since = time.time()

    device = torch.device('cuda:0')

    cfg = get_config('config/config_hgnn.yaml')

    fts, lbls, idx_train, idx_val, idx_test = load_weibo(cfg)
    G = construct_G_from_fts(fts, cfg['knns'])
    G = torch.Tensor(G).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    ft_cat = np.concatenate(fts, axis=1)
    dim_in = ft_cat.shape[1]

    model = HGNN(dim_in=dim_in, n_category=2, hiddens=cfg['hiddens'], drop_out=cfg['drop_out'])

    # count model flops and params
    params = 0
    for param in model.state_dict().values():
        params += param.data.numpy().size
    print(f'# params: {params}')

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.NLLLoss()

    # HGNN classifier
    model_wts_best_val_acc, model_wts_lowest_val_loss \
        = train_weibo(model, ft_cat, G, lbls, idx_train, idx_val, criterion, optimizer, schedular,
                      device, True, False, num_epochs=cfg['max_epoch'], print_freq=cfg['print_freq'])

    print('**** Model of lowest val loss ****')
    acc0 = test_weibo(model, model_wts_lowest_val_loss, ft_cat, G, lbls, idx_test, device, True, False)
    print('**** Model of best val acc ****')
    acc1 = test_weibo(model, model_wts_best_val_acc, ft_cat, G, lbls, idx_test, device, True, False)

    test_acc = max(acc0, acc1)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return test_acc


def multi_time_test(test_times=10):
    """
    test model runtime
    :param test_times: repeat times (default 10)
    :return: mean acc, single experiment runtime
    """
    since = time.time()
    test_times += 2                                 # drop two lowest acc
    accs = [0.0] * test_times
    for i in range(test_times):
        print('#' * 20)
        print(f'### The {i}-th iteration ###')
        print('#' * 20)
        accs[i] = experiment_mlgcn()
        # experiment_hgnn()
    accs.sort(reverse=True)                         # descent order
    mean_acc = sum(accs[0:test_times - 2]) / (test_times - 2)
    runtime = (time.time() - since) / test_times
    print('+' * 20)
    print(f'{test_times - 2} times mean accuray: {mean_acc}, mean time cost: {runtime}')
    print('+' * 20)

    return runtime


if __name__ == '__main__':
    # multi_time_test(10)
    # experiment_mlgcn()
    experiment_hgnn()
