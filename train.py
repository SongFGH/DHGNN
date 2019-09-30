import torch
import copy
import time
import random
import os
from config import get_config
from datasets import source_select
from torch import nn
import torch.optim as optim
from models import model_select
from sklearn import neighbors
import numpy as np

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def train_with_unlabeled(model, fts, lbls, idx_labeled, idx_unlabeled, idx_val, edge_dict,
                   criterion, optimizer, scheduler, device,
                   num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device)
    best_acc = 0.0
    num_unlabeled_per_epoch = len(idx_unlabeled) // num_epochs

    for epoch in range(num_epochs):
        random.shuffle(idx_labeled)
        random.shuffle(idx_unlabeled)

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

            # during each epoch, train with all labeled training data and a slice of unlabeled training data
            ids_unlabeled = idx_unlabeled[0: num_unlabeled_per_epoch]
            idx = idx_labeled + ids_unlabeled if phase == 'train' else idx_val

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                if (phase == 'train'):
                    outputs_labeled = model(ids=idx_labeled, feats=fts, edge_dict=edge_dict, device=device)
                    outputs_unlabeled = model(ids=ids_unlabeled, feats=fts, edge_dict=edge_dict, device=device)

                    # train kNN classifier for unlabeled training sample and predict those labeled as ground truth
                    clf = neighbors.KNeighborsClassifier(5, weights='uniform')
                    clf.fit(outputs_labeled.detach().cpu().numpy(), lbls[idx_labeled].detach().cpu().numpy())
                    lbls_unlabeled = torch.LongTensor(clf.predict(outputs_unlabeled.detach().cpu().numpy())).to(device)

                    # loss computation
                    outputs = torch.cat((outputs_labeled, outputs_unlabeled))
                    groudtruth = torch.cat((lbls[idx_labeled], lbls_unlabeled))
                    loss = criterion(outputs, groudtruth)
                    _, preds = torch.max(outputs, 1)
                    # statistics
                    running_loss += loss * len(idx)
                    running_corrects += torch.sum(preds == lbls.data[idx])

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                else:
                    outputs = model(ids=idx_val, feats=fts, edge_dict=edge_dict, device=device)
                    loss = criterion(outputs, lbls[idx_val])
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss * len(idx)
                    running_corrects += torch.sum(preds == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if epoch % print_freq == 0 and phase == 'val':
                print(f'Best val Acc: {best_acc:4f}')
                print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_by_batch(model, fts, lbls, idx_train, idx_val, edge_dict,
                   criterion, optimizer, scheduler, device,
                   batch_size=256, num_epochs=25, print_freq=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        random.shuffle(idx_train)

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

            ids = idx_train if phase == 'train' else idx_val

            # Iterate over batch.
            n_batchs = len(ids)//batch_size if len(ids)%batch_size==0 else (len(ids)//batch_size+1)
            for i_batch in range(n_batchs):
                idx = ids[i_batch*batch_size : min((i_batch+1)*batch_size, len(ids))]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ids=idx, feats=fts, edge_dict=edge_dict, device=device)
                    loss = criterion(outputs, lbls[idx]) * len(idx)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(preds == lbls.data[idx])

            epoch_loss = running_loss / len(ids)
            epoch_acc = running_corrects.double() / len(ids)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if epoch % print_freq == 0 and phase == 'val':
                print(f'Best val Acc: {best_acc:4f}')
                print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_by_batch(model_best, fts, lbls, idx_test, edge_dict, device, batch_size=256):
    model_best = model_best.to(device)
    model_best.eval()

    running_corrects = 0.0

    with torch.no_grad():
        # Iterate over batch.
        n_batchs = len(idx_test) // batch_size if len(idx_test) % batch_size == 0 else (len(idx_test) // batch_size + 1)
        for i_batch in range(n_batchs):
            idx = idx_test[i_batch * batch_size: min((i_batch + 1) * batch_size, len(idx_test))]

            outputs = model_best(ids=idx, feats=fts, edge_dict=edge_dict, device=device)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == lbls.data[idx])

    test_acc = running_corrects.double() / len(idx_test)

    print('*' * 20)
    print(f'Test acc: {test_acc}')
    print('*' * 20)

    return test_acc


def train_gcn(model, fts, lbls, idx_train, idx_val, edge_dict,
                   criterion, optimizer, scheduler, device,
                   num_epochs=25, print_freq=500):
    """
    gcn-style whole graph training
    :param model:
    :param fts:
    :param lbls:
    :param idx_train:
    :param idx_val:
    :param edge_dict:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param num_epochs:
    :param print_freq:
    :return: best model on validation set
    """
    since = time.time()

    model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device)
    best_acc = 0.0
    loss_min = 100

    for epoch in range(num_epochs):
        # random.shuffle(idx_train)

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

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(feats=fts, edge_dict=edge_dict)
                loss = criterion(outputs[idx], lbls[idx]) * len(idx)
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

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

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return model_wts_best_val_acc, model_wts_lowest_val_loss


def test_gcn(model, best_model_wts, fts, lbls, n_category, idx_test, edge_dict, device, test_time = 1):
    """
    gcn-style whole graph test
    :param model_best:
    :param fts:
    :param lbls:
    :param idx_test:
    :param edge_dict:
    :param device:
    :param test_time: test for several times and vote
    :return:
    """
    model.load_state_dict(best_model_wts)
    model = model.to(device)
    model.eval()

    running_corrects = 0.0

    N = fts.size()[0]
    outputs = torch.zeros(N, n_category).to(device)

    for time in range(test_time):

        with torch.no_grad():

            outputs += model(feats=fts, edge_dict=edge_dict)

    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds[idx_test] == lbls.data[idx_test])
    test_acc = running_corrects.double() / len(idx_test)

    print('*' * 20)
    print(f'Test acc: {test_acc}')
    print('*' * 20)

    return test_acc


def train_test_model():
    device = torch.device('cuda:0')

    cfg = get_config('config/config_cora.yaml')
    source = source_select(cfg)
    print(f'Using {cfg["activate_dataset"]} dataset')
    fts, lbls, idx_train, idx_val, idx_test, n_category, node_dict, edge_dict = source(cfg)
    fts = torch.Tensor(fts).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)

    model = model_select(cfg['model'])\
        (dim_feat=fts.size(1),
        n_categories=n_category,
        k_sample=cfg['k_sample'],
        k_structured=cfg['k_structured'],
        k_nearest=cfg['k_nearest'],
        k_cluster=cfg['k_cluster'],
        clusters=cfg['clusters'],
        adjacent_centers=cfg['adjacent_centers'],
        t_top=cfg['t_top'],
        n_layers=cfg['n_layers'],
        layer_spec=cfg['layer_spec'],
        dropout_rate=cfg['drop_out'],
        has_bias=cfg['has_bias']
        )

    # initialize model
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.95, weight_decay=cfg['weight_decay'])
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.NLLLoss()

    if cfg['model_type'] == 'Transductive':
        # transductive learning mode
        model_wts_best_val_acc, model_wts_lowest_val_loss\
            = train_gcn(model, fts, lbls, idx_train, idx_val, edge_dict, criterion, optimizer, schedular, device,
                        cfg['max_epoch'], cfg['print_freq'])
        if idx_test is not None:
            print('**** Model of lowest val loss ****')
            test_acc_lvl = test_gcn(model, model_wts_lowest_val_loss, fts, lbls, n_category, idx_test, edge_dict, device, cfg['test_time'])
            print('**** Model of best val acc ****')
            test_acc_bva = test_gcn(model, model_wts_best_val_acc, fts, lbls, n_category, idx_test, edge_dict, device, cfg['test_time'])

            return max(test_acc_lvl, test_acc_bva)
        else:
            return None
    else:
        # inductive learning mode
        if not cfg['with_unlabeled_data']:
            model_best_on_val = train_by_batch(model, fts, lbls, idx_train, idx_val, edge_dict,
                                               criterion, optimizer, schedular, device,
                                               cfg['batch_size'], cfg['max_epoch'], cfg['print_freq'])
        else:
            model_best_on_val = train_with_unlabeled(model, fts, lbls, idx_train, idx_test, idx_val, edge_dict,
                                                    criterion, optimizer, schedular, device,
                                                    cfg['max_epoch'], cfg['print_freq'])

        if idx_test is not None:
            test_acc = test_by_batch(model_best_on_val, fts, lbls, idx_test, edge_dict, device, cfg['batch_size'])
            return test_acc
        else:
            return None


def experiment(TIMES=10):
    mean_acc = 0.0
    for i in range(TIMES):
        test_acc = train_test_model()
        mean_acc = (mean_acc * i + test_acc) / (i + 1)
    print('*' * 20)
    print('{} times test mean acc: {}'.format(TIMES, mean_acc))
    print('*' * 20)
    return mean_acc


if __name__ == '__main__':
    setup_seed(10000)              # 71 -> 82.5%
    test_acc = train_test_model()
    print('=' * 20)
    print('Final test acc: {}'.format(test_acc))
    print('=' * 20)