import argparse
import collections
import os

import faiss

import numpy as np

from tqdm import tqdm


def main(args):
    np.random.seed(args.seed)

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k, sparse=args.lookup_sparse)

    reduce_func = ReduceDimensions(args)

    dsize = dstore.keys.shape[0]

    # get splits
    index = np.arange(dsize)
    if args.lookup_sparse:
        sparse_mask = dstore.lookup_done[:].reshape(-1) == 1
        index = index[sparse_mask]
    np.random.shuffle(index)

    splits = [
        ('train', args.ntrain),
        ('vali', args.nvalid),
    ]

    print('mkdir -p {}'.format(args.output))
    os.system('mkdir -p {}'.format(args.output))

    offset = 0
    for name, size in splits:
        path = os.path.join(args.output, '{}.txt'.format(name))
        local_index = index[offset:offset+size]
        tgts = dstore.tgts[local_index]
        knns = dstore.knns[local_index][:, :args.k]
        # dist is sorted from hi to lo
        dist = dstore.dist[local_index][:, :args.k]
        knn_tgts = dstore.tgts[knns.reshape(-1)].reshape(size, args.k, 1)

        # get qids
        qids = local_index

        # get features
        feat = dstore.keys[knns.reshape(-1)].reshape(size, args.k, -1)

        # get label
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

        del tgts
        del knn_tgts
        #del knns

        # get optimal order

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

        new_order = np.zeros((size, args.k, 1)).astype(np.int)
        # set positives
        new_order[positive_dist_sorted > -np.inf] = positive_order[positive_dist_sorted > -np.inf]
        # set negatives
        new_order[negative_dist_sorted > -np.inf] = negative_order[negative_dist_sorted > -np.inf]

        assert np.all(np.unique(new_order, return_counts=True)[1] == size)

        label = np.take_along_axis(label, new_order, axis=1)
        feat = np.take_along_axis(feat, new_order, axis=1)
        knns = np.take_along_axis(knns, new_order, axis=1)
        feat = reduce_func(feat)

        if args.feat_idx:
            print('writing {} with feat_idx shape {}'.format(path, knns.shape))
            write_allrank_data(path, qids, label, None, feat_idx=knns)
        else:
            print('writing {} with feat shape {}'.format(path, feat.shape))
            write_allrank_data(path, qids, label, feat)


class ReduceDimensions:
    def __init__(self, args):
        self.enabled = args.quant
        self.index = faiss.read_index(os.path.join(args.dstore, 'knn.index.trained'))

    def __call__(self, vecs):
        if not self.enabled:
            return vecs
        raise Exception('Do not do this! Product vectors are too much like categorical features.')
        b, k, d = vecs.shape
        new_vecs = self.index.pq.compute_codes(vecs.reshape(b * k, d)) / 255
        return new_vecs.reshape(b, k, -1)


def format_vec(vec):
    if not isinstance(vec, list):
        assert len(vec.shape) == 1
        vec = vec.tolist()
    return ' '.join(['{}:{:.4f}'.format(i, x) for i, x in enumerate(vec)])


def write_allrank_data(path, qids, label, feat, feat_idx=None):
    size, k, _ = label.shape
    with open(path, 'w') as f:
        for i_slate in range(size):
            for i_k in range(k):
                q = int(qids[i_slate])
                y = int(label[i_slate, i_k, 0])
                if feat_idx is None:
                    val = format_vec(feat[i_slate, i_k])
                    f.write('{} qid:{} {}'.format(y, q, val))
                else:
                    val = int(feat_idx[i_slate, i_k, 0])
                    f.write('{} qid:{} 0:{}'.format(y, q, val))
                f.write('\n')


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self):
        path = self.path
        self.keys = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.tgts = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.prob = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        self._initialized = True

    def add_neighbors(self, path, k, sparse=False):
        self.knns = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))
        if sparse:
            self.lookup_done = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='dstore_train', type=str)
    parser.add_argument('--dstore-size', default=103225485, type=int)
    # dstore neighbors
    parser.add_argument('--lookup', default='lookup_train', type=str)
    parser.add_argument('--lookup-k', default=128, type=int)
    parser.add_argument('--lookup-sparse', action='store_true')
    # allrank
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--ntrain', default=1000, type=int)
    parser.add_argument('--nvalid', default=100, type=int)
    parser.add_argument('--seed', default=121, type=int)
    parser.add_argument('--k', default=128, type=int)
    parser.add_argument('--feat-idx', action='store_true')
    parser.add_argument('--quant', action='store_true')
    args = parser.parse_args()

    print(args)

    main(args)

