#!/usr/bin/env python
# coding=utf-8
# Created Time:    2018-08-07 16:31:55
# Modified Time:   2018-08-27 10:34:32
import numpy as np

def mAP_func(features, labels, topk=1000):
    first = np.expand_dims(features, 1)
    second = features
    dist = (first-second)
    dist = (dist*dist).sum(-1)
    ind = dist.argsort(1)[:,:topk]
    tmp = (labels[ind]==np.expand_dims(labels, 1)).astype(np.int)
    tmp = np.expand_dims(labels, 0)
    mat = tmp.cumsum(1)
    di = np.ones((1,topk)).cumsum(-1)
    pr = mat/di
    for i in range(topk):
        pr[:, i] = np.max(pr[:,i:],-1, keepdims=True)

    rslt = ((pr*tmp).sum(1)/(tmp.sum(1))).mean()
    return rslt

def main():
    scores = np.array([0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11, 0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.1, 0.23, 0.46, 0.08])
    gt_label = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]              )
    scores = np.expand_dims(scores, 1)
    print(mAP_func(scores, gt_label, 20))
if __name__ == "__main__":
    main()
