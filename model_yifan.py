import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_cluster import knn_graph
from torch_sparse import coalesce
from torch_scatter import scatter_mean


def get_edge_index_from_dict(edge_dict: list, device):
    edge_index = []
    for idx, vs in enumerate(edge_dict):
        for v in vs:
            edge_index.append([idx, v])
    edge_index = torch.tensor(edge_index).long().to(device).permute(1, 0)
    return edge_index


def get_edge_dict_from_index(edge_index: torch.tensor):
    edge_index = edge_index.cpu().data.numpy()
    edge_dict = [[] for _ in range(edge_index[0].max() + 1)]
    for idx in range(edge_index.shape[1]):
        x, y = edge_index[:, idx]
        edge_dict[x].append(y)
    return edge_dict


def sample_neighbors(edge_index, neig_k):
    """
    Samples k neighbors, returns neighbors' indices matrix
    :param edge_index: (2, num_edge)
    :param neig_k:
    :return: (N, neig_k)
    """
    row_len, col_len = edge_index[0].max() + 1, edge_index[1].max() + 1
    edge_index, _ = coalesce(edge_index, None, row_len, col_len)
    edge_index_numpy = edge_index.cpu().data.numpy()
    edge_dict = defaultdict(list)
    for idx in range(edge_index_numpy.shape[1]):
        edge_dict[edge_index_numpy[0, idx]].append(edge_index_numpy[1, idx])
    edge_neighbor_index = torch.zeros((row_len, neig_k)).long()  # (E, k)
    for row_i in edge_dict:
        edge_neighbor_index[row_i] = torch.from_numpy(np.random.choice(edge_dict[row_i], neig_k)).long()
    return edge_neighbor_index.to(edge_index.device)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class Transform(torch.nn.Module):
    """
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """

    def __init__(self, dim_in, neig_k):
        super(Transform, self).__init__()

        if (neig_k * neig_k) % dim_in == 0:
            self.convKK = torch.nn.Conv1d(dim_in, neig_k * neig_k, neig_k, groups=dim_in)
        elif dim_in % (neig_k * neig_k) == 0:
            self.convKK = torch.nn.Conv1d(dim_in, neig_k * neig_k, neig_k, groups=neig_k * neig_k)
        else:
            self.convKK = torch.nn.Conv1d(dim_in, neig_k * neig_k, neig_k)

        self.act = torch.nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.convKK.weight)
        zeros(self.convKK.bias)

    def forward(self, region_feats: torch.tensor):
        N, neig_k, d = region_feats.size()  # (N, k, d)
        region_feats = region_feats.permute(0, 2, 1)  # (N, d, k)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, neig_k, neig_k)  # (N, k, k)
        multiplier = self.act(multiplier)
        region_feats = region_feats.permute(0, 2, 1)  # (N, k, d)
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats


class ConvMapping(torch.nn.Module):

    def __init__(self, dim_in, neig_k):
        super(ConvMapping, self).__init__()
        self.neig_k = neig_k
        self.trans = Transform(dim_in, neig_k)
        self.convK1 = torch.nn.Conv1d(neig_k, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.convK1.weight)
        zeros(self.convK1.bias)

    def forward(self, x, neigs_index):
        neighs_size = neigs_index.size()
        neigs_index = neigs_index.view(-1)  # (N * k, )

        region_feats = torch.index_select(x, dim=0, index=neigs_index)  # (N * k, d)
        region_feats = region_feats.view(*neighs_size, -1)  # (N, k, d)

        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)
        pooled_feats = pooled_feats.squeeze()
        return pooled_feats


class SelfAttention(torch.nn.Module):
    """
    Adjacent clusters self attention layer
    """

    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.att_net = torch.nn.Sequential(
            torch.nn.Linear(dim_ft, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Softmax(dim=-1)
        )
        # self.fc_0 = torch.nn.Linear(dim_ft, hidden)
        # self.fc_1 = torch.nn.Linear(hidden, 1)
        # self.act_0 = torch.nn.ReLU()
        # self.act_1 = torch.nn.Softmax(dim=-1)

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        # att = self.act_1(self.fc_1(self.act_0(self.fc_0(ft))))      # (N, t, 1)
        att = self.att_net(ft)  # (N, t, 1)
        return torch.sum(att * ft, dim=-2).squeeze()


class DHGNNRawConv(torch.nn.Module):
    """The hypergraph convolution operation from the `"Hypergraph Neural Networks"`"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 neig_s,
                 neig_k,
                 bias=True):
        super(DHGNNRawConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neig_k = neig_k
        self.neig_s = neig_s
        self.cached_result = None

        self.trans_s = ConvMapping(out_channels, neig_s)
        self.trans_k = ConvMapping(out_channels, neig_k)
        self.self_att = SelfAttention(out_channels, out_channels // 4)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)  # (N, d)

        # raw structure
        edge_neighs_index = sample_neighbors(edge_index, self.neig_s)  # (N, k)
        # for node_idx in range(edge_neighs_index.size(0)):
        #     assert set(edge_neighs_index[node_idx].cpu().data.numpy()).\
        #         issubset(edge_index[1, edge_index[0]==node_idx].cpu().data.numpy()), \
        #         f'fetch error in node {node_idx}!'
        x_s = self.trans_s(x, edge_neighs_index).unsqueeze(1)

        # # knn structure
        knn_edge_index = knn_graph(x, self.neig_k)
        knn_neighs_index = sample_neighbors(knn_edge_index, self.neig_k)  # (N, k)
        x_k = self.trans_k(x, knn_neighs_index).unsqueeze(1)  # (N, d)

        x = torch.cat((x_s, x_k), dim=1)
        # x = x_s.squeeze()

        x = self.self_att(x)

        if self.bias is not None:
            return x + self.bias
        else:
            return x

class GCNConv(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(GCNConv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.weight)
        x_j = torch.index_select(x, 0, edge_index[1])
        out = scatter_mean(x_j, edge_index[0], 0)
        if self.bias is not None:
            out = out + self.bias
        return out


class Net(torch.nn.Module):

    def __init__(self,
                 feat_dim,
                 hidden_dim,
                 n_class,
                 neig_s,
                 neig_k):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        # self.conv1 = DHGNNRawConv(feat_dim, hidden_dim, neig_s=neig_s, neig_k=neig_k)
        self.conv2 = DHGNNRawConv(hidden_dim, n_class, neig_s=neig_s, neig_k=neig_k)

    def forward(self, feats, edge_dict):
        x, edge_index = feats, edge_dict
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
