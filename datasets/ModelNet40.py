import scipy.io as scio
import numpy as np
from utils import construct_hypergraph as ch


def load_ft(data_dir, feature_name='GVCNN'):
    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test


def load_modelnet40_data(cfg):
    # init feature
    mvcnn_ft, lbls, idx_train, idx_val = load_ft(cfg['modelnet40_ft'], feature_name='MVCNN')
    gvcnn_ft, lbls, idx_train, idx_val = load_ft(cfg['modelnet40_ft'], feature_name='GVCNN')
    fts = None
    if 'mvcnn_ft' not in dir() or 'gvcnn_ft' not in dir():
        raise Exception('None feature initialized')
    if cfg['use_mvcnn_feature_for_train'] and cfg['use_gvcnn_feature_for_train']:
        fts = np.hstack((mvcnn_ft, gvcnn_ft))
    elif cfg['use_mvcnn_feature_for_train']:
        fts = mvcnn_ft
    elif cfg['use_gvcnn_feature_for_train']:
        fts = gvcnn_ft
    if fts is None:
        raise Exception('None feature for training!')
    print(f'feature dimension: {fts.shape}')

    # construct hypergraph incidence matrix
    H = None
    if cfg['use_mvcnn_feature_for_structure']:
        tmp = ch.construct_H_with_KNN(mvcnn_ft, K_neigs=cfg['K_neigs'],
                                        is_probH=False, m_prob=1)
        H = ch.hyperedge_concat(H, tmp)
    if cfg['use_gvcnn_feature_for_structure']:
        tmp = ch.construct_H_with_KNN(gvcnn_ft, K_neigs=cfg['K_neigs'],
                                        is_probH=False, m_prob=1)
        H = ch.hyperedge_concat(H, tmp)
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')
    print(f'graph dimension: {H.shape}')

    # construct node_dict & edge_dict from H
    node_dict, edge_dict = ch.H_to_node_edge_dict(H)

    n_category = lbls.max() - lbls.min() + 1

    return fts, lbls, idx_train, idx_val, None, n_category, node_dict, edge_dict
