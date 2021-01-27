import argparse
import collections
import os

import faiss

import numpy as np

from tqdm import tqdm


def main(args):
    np.random.seed(args.seed)

    os.system('mkdir -p {}'.format(args.output))

    print('TRAIN')
    out = build_split(args.tr_dstore, args.tr_dstore_size, args.tr_lookup, args.tr_lookup_k, args.k, args.ntrain, shuffle=True, balance=True)
    path = os.path.join(args.output, 'train.txt')
    write_allrank_data(path, out['qids'], out['label'], out['feat_idx'])

    print('VALID')
    out = build_split(args.va_dstore, args.va_dstore_size, args.va_lookup, args.va_lookup_k, args.k, args.nvalid, shuffle=False, balance=False)
    path = os.path.join(args.output, 'vali.txt')
    write_allrank_data(path, out['qids'], out['label'], out['feat_idx'])


def build_split(split_dstore_path, split_dstore_size, lookup_path, lookup_k, k, n, shuffle=True, balance=False):
    split_dstore = Dstore(split_dstore_path, split_dstore_size, 1024)
    split_dstore.initialize(has_row=True)
    split_dstore.add_neighbors(lookup_path, lookup_k)

    index = np.arange(split_dstore.tgts.shape[0])
    if shuffle:
        np.random.shuffle(index)
    index = index[:n]

    tgts = split_dstore.tgts[index]
    knns = split_dstore.knns[index, :lookup_k]
    dist = split_dstore.dist[index, :lookup_k] # dist is sorted from hi to lo
    knn_tgts = split_dstore.knn_tgts[index, :lookup_k]
    qids = split_dstore.row[index]
    size = qids.shape[0]
    label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    if balance:
        npos = k // 3
        nneg = k - npos

        # count stats
        print('STATS (BEFORE BALANCE)')
        has_positive = label.reshape(size, lookup_k).sum(axis=1) >= npos
        has_negative = label.reshape(size, lookup_k).sum(axis=1) < lookup_k - nneg
        has_both = np.logical_and(has_positive, has_negative)
        print('has enough positive = {} / {}'.format(np.sum(has_positive), size))
        print('has enough negative = {} / {}'.format(np.sum(has_negative), size))
        print('has_both = {} / {}'.format(np.sum(has_both), size))

        order = np.arange(lookup_k).reshape(1, -1).repeat(size, axis=0).reshape(size, lookup_k, 1).astype(np.int)

        pos_order = order.copy()
        pos_order[label == 0] = lookup_k + 10**6
        pos_order.sort(axis=1)
        pos_order = pos_order[:, :npos]
        assert pos_order[has_both].max() < lookup_k

        neg_order = order.copy()
        neg_order[label == 1] = lookup_k + 10**6
        neg_order.sort(axis=1)
        neg_order = neg_order[:, :nneg]
        assert neg_order[has_both].max() < lookup_k

        def _filter(x, i0, i1):
            x0 = np.take_along_axis(x[has_both], i0[has_both], axis=1)
            x1 = np.take_along_axis(x[has_both], i1[has_both], axis=1)
            return np.concatenate([x0, x1], axis=1)

        tgts = tgts[has_both]
        knns = _filter(knns, pos_order, neg_order)
        dist = _filter(dist, pos_order, neg_order)
        knn_tgts = _filter(knn_tgts, pos_order, neg_order)
        qids = qids[has_both]
        size = qids.shape[0]
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)


    else:
        knns = knns[:, :k]
        dist = dist[:, :k]
        knn_tgts = knn_tgts[:, :k]
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    # count stats
    print('STATS')
    has_positive = label.reshape(size, k).sum(axis=1) > 0
    has_negative = label.reshape(size, k).sum(axis=1) < k
    has_both = np.logical_and(has_positive, has_negative)
    print('has_positive = {} / {}'.format(np.sum(has_positive), size))
    print('has_negative = {} / {}'.format(np.sum(has_negative), size))
    print('has_both = {} / {}'.format(np.sum(has_both), size))

    del tgts
    del knn_tgts


    qids = qids[has_both]
    dist = dist[has_both]
    label = label[has_both]
    knns = knns[has_both]

    size = label.shape[0]

    # get optimal order
    print('get optimal order')

    ## positives - sort from hi to lo
    positive_dist = dist.copy()
    positive_dist[label == 0] = -np.inf
    positive_dist_sorted = np.sort(positive_dist, axis=1)[:, ::-1]
    positive_order = positive_dist.argsort(axis=1)[:, ::-1]

    ## negatives - sort from lo to hi
    negative_dist = dist.copy()
    negative_dist[label == 1] = -np.inf
    negative_dist_sorted = np.sort(negative_dist, axis=1)
    negative_order = negative_dist.argsort(axis=1)

    # set positives and negatives
    new_order = np.zeros((size, k, 1)).astype(np.int)
    new_order[positive_dist_sorted > -np.inf] = positive_order[positive_dist_sorted > -np.inf]
    new_order[negative_dist_sorted > -np.inf] = negative_order[negative_dist_sorted > -np.inf]

    assert np.all(np.unique(new_order, return_counts=True)[1] == size)

    print('re-order')
    label = np.take_along_axis(label, new_order, axis=1)
    knns = np.take_along_axis(knns, new_order, axis=1)

    out = {}
    out['qids'] = qids
    out['label'] = label
    out['feat_idx'] = knns

    return out


def write_allrank_data(path, qids, label, feat_idx):
    print('writing {} with feat_idx shape {}'.format(path, feat_idx.shape))
    size, k, _ = label.shape
    with open(path, 'w') as f:
        for i_slate in range(size):
            for i_k in range(k):
                q = int(qids[i_slate])
                y = int(label[i_slate, i_k, 0])
                f.write('{} qid:{} 0:{}'.format(y, q, feat_idx[i_slate, i_k, 0]))
                f.write('\n')


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self, has_row=False):
        path = self.path
        self.keys = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.tgts = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.prob = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        if has_row:
            self.row = np.memmap(os.path.join(path, 'dstore_row.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self._initialized = True

    def add_neighbors(self, path, k):
        self.knns = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.knn_tgts = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))
        self.lookup_done = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dstore
    #parser.add_argument('--dstore', default='dstore_train', type=str)
    #parser.add_argument('--dstore-size', default=103225485, type=int)
    # dstore-tr
    parser.add_argument('--tr-dstore', default='from_dstore_valid/tr', type=str)
    parser.add_argument('--tr-dstore-size', default=100000, type=int)
    parser.add_argument('--tr-lookup', default='from_dstore_valid/lookup_tr', type=str)
    parser.add_argument('--tr-lookup-k', default=1024, type=int)
    parser.add_argument('--ntrain', default=5000, type=int)
    # dstore-va
    parser.add_argument('--va-dstore', default='from_dstore_valid/va', type=str)
    parser.add_argument('--va-dstore-size', default=10000, type=int)
    parser.add_argument('--va-lookup', default='from_dstore_valid/lookup_va', type=str)
    parser.add_argument('--va-lookup-k', default=1024, type=int)
    parser.add_argument('--nvalid', default=10000, type=int)
    parser.add_argument('--va-shuffle', action='store_true')
    # allrank
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--seed', default=121, type=int)
    parser.add_argument('--k', default=32, type=int)
    args = parser.parse_args()

    print(args)

    main(args)

