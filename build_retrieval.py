import json
import os

import numpy as np
from tqdm import tqdm

from knnlm import KNN_Index
from dstore_util import *
from dataset_util import *


def main(args):
    print('load train')
    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=True)
    vals = npy_copy(knn_dstore.vals[:])

    print('load eval')
    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize(include_keys=True)

    print('load index')
    k = args.k
    nprobe = 32
    path = args.index
    index = KNN_Index(path, nprobe, k)

    q = npy_copy(dstore.keys[:])
    if args.limit > 0:
        q = q[:args.limit]

    print('query')
    dist, knns = index.get_knns(q)

    print('get vals')
    u, inverse = np.unique(knns, return_inverse=True)
    u_tgts = vals[u]
    knn_tgts = u_tgts[inverse].reshape(-1, k)

    print('writing...')
    n = q.shape[0]
    tmp = np.memmap(os.path.join(args.output, 'lookup_knns.npy'), dtype=np.int, mode='w+', shape=(n, k))
    tmp[:] = knns
    tmp = np.memmap(os.path.join(args.output, 'lookup_knn_tgts.npy'), dtype=np.int, mode='w+', shape=(n, k))
    tmp[:] = knn_tgts
    tmp = np.memmap(os.path.join(args.output, 'lookup_dist.npy'), dtype=np.float32, mode='w+', shape=(n, k))
    tmp[:] = dist

    if args.exact:
        def dist_func(a, b):
            return np.sum((a - b)**2, axis=-1)

        batch_size = 32

        tmp = np.memmap(os.path.join(args.output, 'lookup_exact.npy'), dtype=np.float32, mode='w+', shape=(n, k))

        for start in tqdm(range(0, n, batch_size), desc='exact'):
            end = min(start + batch_size, n)
            q_ = q[start:end].reshape(-1, 1, 1024)
            knns_ = knns[start:end].reshape(-1)
            keys = knn_dstore.keys[knns_].reshape(-1, k, 1024)
            d = dist_func(q_, keys)
            tmp[start:end] = d.reshape(-1, k)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    #
    parser.add_argument('--index', default='filtered_dstore_train:keep_non_active=false/knn.index')
    # dstore
    parser.add_argument('--knn-dstore', default='dstore_train', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    #
    parser.add_argument('--k', default=1024, type=int)
    #
    parser.add_argument('--limit', default=-1, type=int)
    # output
    parser.add_argument('--output', default='filtered_dstore_train:keep_non_active=false/lookup')
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()

    # Print flags.
    print(args)

    os.system('mkdir -p {}'.format(args.output))

    main(args)

