import numpy as np
import scipy.io as scio
from config import get_config
from random import sample


def load_weibo_ft(data_dir):
    data = scio.loadmat(data_dir)
    data = data['mImageTextFeaSel']
    text_ft, emojicon_ft, visual_ft = \
        data[0][0].astype(np.double), \
        data[1][0].astype(np.double), \
        data[2][0].astype(np.double)
    return text_ft, emojicon_ft, visual_ft


def load_weibo_sentiment(data_dir):
    data = scio.loadmat(data_dir)
    data = data['mWordSentimentCell']
    text_senti, emojicon_senti, visual_senti = \
        data[0][0][:, 0].astype(np.double),\
        data[1][0][:, 0].astype(np.double),\
        data[2][0][:, 0].astype(np.double)
    return text_senti, emojicon_senti, visual_senti


def load_weibo(cfg):
    text_ft, emojicon_ft, visual_ft = load_weibo_ft(cfg['weibo_ft'])			# /path/of/mImageTextFeaSel.mat
    text_st, emojicon_st, visual_st = load_weibo_sentiment(cfg['weibo_sentiment'])	# /path/of/mWordSentimentCell_2547_49_1553.mat
    text_ft *= text_st
    emojicon_ft *= emojicon_st
    visual_ft *= visual_st
    fts = [text_ft, emojicon_ft, visual_ft]
    lbls = [1 for i in range(4196)] + [0 for i in range(4196, 5550)]      # 1: positive, 0: negative
    # random select test set
    idx_test = sample(range(4196), 378) + sample(range(4196, 5550), 122)
    idx_train_val = list(set(range(5550)) - set(idx_test))
    idx_val = sample(idx_train_val, 400)
    idx_train = list(set(idx_train_val) - set(idx_val))
    return fts, lbls, idx_train, idx_val, idx_test


if __name__ == '__main__':
    cfg = get_config('../config/config.yaml')						# /path/of/dataset
    fts = load_weibo(cfg)
    print(fts)
