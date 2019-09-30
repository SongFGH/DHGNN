# External Modules
import torch
from torch import cuda, FloatTensor, LongTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from typing import Union
import random
# Types to allow for both CPU and GPU models.
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]

def get_indices(batch_size, sample_num=1024, point_num=2048, pool_setting=None):
    if not isinstance(point_num, np.ndarray):
        point_nums = np.full((batch_size), point_num)
    else:
        point_nums = point_num

    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        if pool_setting is None:
            pool_size = pt_num
        else:
            if isinstance(pool_setting, int):
                pool_size = min(pool_setting, pt_num)
            elif isinstance(pool_setting, tuple):
                pool_size = min(random.randrange(pool_setting[0], pool_setting[1]+1), pt_num)
        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                      np.random.choice(pool_size, sample_num - pool_size, replace=True)))
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]
        choices = np.expand_dims(choices, axis=1)
        choices_2d = np.concatenate((np.full_like(choices, i), choices), axis=1)
        indices.append(choices_2d)
    return np.stack(indices)

def knn_indices_func_cpu(rep_pts : FloatTensor,  # (N, pts, dim)
                         pts : FloatTensor,      # (N, x, dim)
                         K : int, D : int
                        ) -> LongTensor:         # (N, pts, K)
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis = 0))
    return region_idx

def knn_indices_func_gpu(rep_pts : cuda.FloatTensor,  # (N, pts, dim)
                         pts : cuda.FloatTensor,      # (N, x, dim)
                         k : int, d : int
                        ) -> cuda.LongTensor:         # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []

    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref)**2, 2).squeeze()
        _, inds = torch.topk(dist2, k*d + 1, dim = 1, largest = False)
        region_idx.append(inds[:,1::d])

    region_idx = torch.stack(region_idx, dim = 0)
    return region_idx
