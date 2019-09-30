import h5py
import numpy as np
import os
import os.path as osp
# import matplotlib.pyplot as plt
from sklearn import metrics

# all 2468 shapes
top_k = 1000

def dist_fun(a, b):
    return np.linalg.norm(a - b)

def cal_dist(features):
    num = len(features)
    dist_mat = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(i, num):
            tmp = dist_fun(features[i], features[j])
            dist_mat[i][j] = tmp
            dist_mat[j][i] = tmp
    return dist_mat

def read_cal_pr(des_mat, labels, save = True, draw = False):
    num = len(labels)
    precisions = []
    recalls = []
    ans = []
    for i in range(num):
        scores = des_mat[:, i]
        targets = (labels == labels[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        tmp = 0
        sum = truth[:top_k].sum()
        precision = []
        recall = []
        for j in range(top_k):
            if truth[j]:
                tmp+=1
                # precision.append(sum/(j + 1))
            recall.append(tmp*1.0/sum)
            precision.append(tmp*1.0/(j+1))
        precisions.append(precision)
        for j in range(len(precision)):
            precision[j] = max(precision[j:])
        recalls.append(recall)
        tmp = []
        for ii in range(11):
            min_des = 100
            val = 0
            for j in range(top_k):
                if abs(recall[j] - ii * 0.1) < min_des:
                    min_des = abs(recall[j] - ii * 0.1)
                    val = precision[j]
            tmp.append(val)
        ans.append(tmp)
    ans = np.array(ans).mean(0)
    if save:
        save_dir = os.path.join('.', 'pr.csv')
        np.savetxt(save_dir, np.array(ans), fmt='%.3f', delimiter=',')
    # if draw:
    #     plt.plot(ans)
    #     plt.show()
    return ans

def mAP_func(features, labels):
    dist_mat = cal_dist(features)
    num = len(labels)
    mAP = 0
    auc = 0.
    for i in range(num):
        scores = dist_mat[:, i]
        targets = (labels == labels[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        np_tp = np.array(truth)
        t_sum = np_tp.sum()
        f_sum = top_k - t_sum
        tpr = np_tp.cumsum() / t_sum
        fpr = (1 - np_tp).cumsum() / f_sum
        sum = 0
        precision = []
        for j in range(top_k):
            if truth[j]:
                sum+=1
                precision.append(sum*1.0/(j + 1))
        if len(precision) == 0:
            ap = 0
        else:
            for ii in range(len(precision)):
                precision[ii] = max(precision[ii:])
            ap = np.array(precision).mean()
        mAP += ap
        auc += metrics.auc(fpr, tpr)
    mAP = mAP/num
    auc = auc/num
    ans = read_cal_pr(dist_mat, labels, False, False)
    return mAP, auc, ans


def demo_index(features, labels):
    dist_mat = cal_dist(features)
    num = len(labels)
    ret = []
    for i in range(num):
        scores = dist_mat[:, i]
        targets = (labels == labels[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:11]
        truth = targets[sortind]
        ret.append((sortind, truth))
    return ret


def _test():
    scores = [0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11, 0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.1, 0.23, 0.46, 0.08]
    gt_label = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    scores = np.array(scores)
    targets = np.array(gt_label).astype(np.uint8)
    sortind = np.argsort(scores, 0)[::-1]
    truth = targets[sortind]
    sum = 0
    precision = []
    for j in range(20):
        if truth[j]:
            sum += 1
            precision.append(sum*1.0 / (j + 1))
    if len(precision) == 0:
        ap = 0
    else:
        for i in range(len(precision)):
            precision[i] = max(precision[i:])
        ap = np.array(precision).mean()
    print(ap)


if __name__ == '__main__':
    # draw = True
    # cfg = config.config()
    # des_mat, labels = cal_des(cfg)
    # read_cal_map(cfg, des_mat, labels)
    # read_cal_pr(cfg, des_mat, labels, draw= draw)
    _test()
