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
import torch.nn.functional as F
from model_yifan import Net, get_edge_index_from_dict
from planetoid_yifan import read_planetoid_data
from sklearn import neighbors

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


device = torch.device('cuda:0')

cfg = get_config('config/config_cora.yaml')
source = source_select(cfg)
print(f'Using {cfg["activate_dataset"]} dataset')

# yuxuan
fts, lbls, idx_train, idx_val, idx_test, n_category, node_dict, edge_dict = source(cfg)
#
# yifan
# fts, lbls, idx_train, idx_val, idx_test, n_category, node_dict, edge_dict = read_planetoid_data(cfg)


fts = torch.Tensor(fts).to(device)
lbls = torch.tensor(lbls).squeeze().long().to(device)


# yuxuan
# model = model_select(cfg['model'])\
#     (dim_feat=fts.size(1),
#     n_categories=n_category,
#     k_sample=cfg['k_sample'],
#     k_structured=cfg['k_structured'],
#     k_nearest=cfg['k_nearest'],
#     k_cluster=cfg['k_cluster'],
#     clusters=cfg['clusters'],
#     adjacent_centers=cfg['adjacent_centers'],
#     t_top=cfg['t_top'],
#     n_layers=cfg['n_layers'],
#     layer_spec=cfg['layer_spec'],
#     dropout_rate=cfg['drop_out'],
#     has_bias=cfg['has_bias']
#     )

# yifan rewrite
model = Net(feat_dim=fts.size(1),
            hidden_dim=cfg['layer_spec'][0],
            n_class=n_category,
            neig_s=cfg['k_nearest'],
            neig_k=cfg['k_nearest'])
edge_dict = get_edge_index_from_dict(edge_dict, device)

# initialize model
state_dict = model.state_dict()
for key in state_dict:
    if 'weight' in key:
        nn.init.xavier_uniform_(state_dict[key])
    elif 'bias' in key:
        state_dict[key] = state_dict[key].zero_()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])
# optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.95, weight_decay=cfg['weight_decay'])
schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=cfg['milestones'],
                                           gamma=cfg['gamma'])
criterion = torch.nn.NLLLoss()


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(feats=fts, edge_dict=edge_dict)[idx_train], lbls[idx_train]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(feats=fts, edge_dict=edge_dict), []
    for idx in [idx_train, idx_val, idx_test]:
        pred = logits[idx].max(1)[1]
        acc = pred.eq(lbls[idx]).sum().item() / len(lbls[idx])
        accs.append(acc)
    return accs


if __name__ == '__main__':
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = f'Epoch: {epoch:03d}, Train:{train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}'
        print(log)