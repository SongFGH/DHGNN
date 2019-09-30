import os.path as osp
import sys
import numpy as np
from itertools import repeat

import torch
from torch_sparse import coalesce
from model_yifan import get_edge_dict_from_index

import scipy.sparse as sp

# from torch_cluster import knn_graph


try:
    import cPickle as pickle
except ImportError:
    import pickle


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def hyperedge_cat(edge_indices):
    if isinstance(edge_indices, tuple) or isinstance(edge_indices, list):
        edge_indices = [edge_index.clone() for edge_index in edge_indices]
        edge_min_list = []
        edge_max_list = []
        edge_num_list = []
        edge_new_list = []
        for edge_index in edge_indices:
            edge_min_list.append(edge_index[1].min().item())
            edge_max_list.append(edge_index[1].max().item())
            edge_num_list.append(edge_max_list[-1] - edge_min_list[-1] + 1)

        last_edge_num = 0
        for i, edge_index in enumerate(edge_indices):
            edge_index[1, :] = edge_index[1, :] - edge_min_list[i] + last_edge_num
            edge_new_list.append(edge_index)
            last_edge_num += edge_num_list[i]
        edge_new_list = torch.cat(edge_new_list, dim=1)
        return edge_new_list
    else:
        return edge_indices


def remove_self_loops(edge_index, edge_attr=None):
    """
    Removes every self-loop in the graph given by :attr:`edge_index`
    :param edge_index: : The edge indices
    :param edge_attr: Edge weights or multi-dimensional edge features. (default: obj:`None`)
    :return: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.unsqueeze(0).expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


def add_self_loops(edge_index, num_nodes=None):
    """Add self-loop"""
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)
    return edge_index


# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features

def preprocess(features):
    return features / features.sum(1, keepdim=True).clamp(min=1)


def read_planetoid_data(cfg):
    folder, prefix, structure = cfg['citation_root'], cfg['activate_dataset'], 'graph'
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_index = test_index.sort()[0]

    if prefix.lower() == 'citeseer':
        len_test_indices = (test_index.max() - test_index.min())

        tx_ext = torch.zeros(len_test_indices, tx.size(1))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]

    x = preprocess(x)

    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = add_self_loops(edge_index)
    if structure.lower() == 'hypergraph':
        graph_edge_index = edge_index.clone()
        # extra_edges = knn_graph(x.cuda(), 5).cpu()
        # edge_index, _ = convert_graph_to_hypergraph(edge_index)
        edge_index = hyperedge_cat((graph_edge_index, edge_index))
        # edge_index = hyperedge_cat((graph_edge_index, extra_edges))
        # edge_index = hyperedge_cat((edge_index, graph_edge_index))
        edge_index, _ = coalesce(edge_index, None,
                                 edge_index[0].max() + 1,
                                 edge_index[1].max() + 1)

    edge_dict = get_edge_dict_from_index(edge_index)

    return x, y, train_index, val_index, test_index, y.max().item()+1, edge_dict, edge_dict


def read_file(folder, prefix, name):
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes,), dtype=torch.uint8)
    mask[index] = 1
    return mask
