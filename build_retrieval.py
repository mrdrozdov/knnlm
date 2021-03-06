import json
import os

import numpy as np
import torch
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

    q = npy_copy(dstore.keys[:])
    if args.limit > 0:
        q = q[:args.limit]
    n = q.shape[0]

    use_cuda = True
    device = torch.cuda.current_device() if use_cuda else False

    if args.skip_lookup:
        print('restore')
        k = args.k
        knns = np.memmap(os.path.join(args.output, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(n, k))
        knns = npy_copy(knns[:])
        knn_tgts = np.memmap(os.path.join(args.output, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(n, k))
        knn_tgts = npy_copy(knn_tgts[:])

        print('get vals')
        #u, inverse = np.unique(knns, return_inverse=True)
        pt_knns = torch.from_numpy(knns).to(device)
        pt_u, pt_inverse = torch.unique(pt_knns, return_inverse=True, sorted=True)
        u = pt_u.cpu().numpy()
        inverse = pt_inverse.cpu().numpy().reshape(-1)

        del pt_knns, pt_u, pt_inverse

    else:
        print('load index')
        k = args.k
        nprobe = 32
        path = args.index
        index = KNN_Index(path, nprobe, k)

        print('query')
        dist, knns = index.get_knns(q)

        print('get vals')
        u, inverse = np.unique(knns, return_inverse=True)
        u_tgts = vals[u]
        knn_tgts = u_tgts[inverse].reshape(-1, k)

        print('writing...')
        tmp = np.memmap(os.path.join(args.output, 'lookup_knns.npy'), dtype=np.int, mode='w+', shape=(n, k))
        tmp[:] = knns
        tmp = np.memmap(os.path.join(args.output, 'lookup_knn_tgts.npy'), dtype=np.int, mode='w+', shape=(n, k))
        tmp[:] = knn_tgts
        tmp = np.memmap(os.path.join(args.output, 'lookup_dist.npy'), dtype=np.float32, mode='w+', shape=(n, k))
        tmp[:] = -dist

    if args.exact:

        def dist_func(a, b):
            a = torch.from_numpy(a).to(device)
            b = torch.from_numpy(b).to(device)
            d = torch.sum((a - b)**2, axis=-1)
            return d.cpu().numpy()
            #return np.sum((a - b)**2, axis=-1)

        def fancy_lookup(keys, u):
            start, end = u.min(), u.max() + 1

            if end - start > 10**6:
                raise Exception('This is too chunky.')

            chunk = keys[start:end][:]
            return chunk[u - start]

        print('computing exact distance for {} queries and {} keys.'.format(q.shape[0], u.shape[0]))

        batch_size = args.batch_size

        tmp = np.memmap(os.path.join(args.output, 'lookup_exact.npy'), dtype=np.float32, mode='w+', shape=(n, k))

        query_ids = np.arange(n).repeat(k)

        #for start in tqdm(range(0, u.shape[0], batch_size), desc='exact'):
        N = knn_dstore.keys.shape[0]
        u_range = np.arange(u.shape[0])
        u_offset = 0
        for start in tqdm(range(0, N, batch_size), desc='exact'):
            end = min(start + batch_size, N)

            # early exit
            if u_offset >= u.shape[0]:
                break

            # select u start
            u_start = u_offset
            if start > u[u_start]:
                continue
            assert u[u_start] >= start

            # select u end
            u_end = u_start
            for i in range(u_start, u.shape[0]):
                if u[u_offset + i] >= end:
                    break
                u_end = u_offset + i
            u_offset = u_end + 1
            assert u[u_end] < end

            # mask
            mask = np.logical_and(inverse >= u_start, inverse <= u_end).reshape(-1)
            batch_inv = inverse[mask] - u_start
            assert batch_inv.min() == 0
            assert batch_inv.max() == u_end - u_start, (u_end - u_start, batch_inv.max())

            # keys
            batch_u = u[u_start:u_end+1]
            #batch_u_k = knn_dstore.keys[batch_u]
            #batch_u_k = fancy_lookup(knn_dstore.keys, batch_u)

            # fancy
            keys = knn_dstore.keys[start:end][:]
            batch_u_k = keys[batch_u - start]
            batch_k = batch_u_k[batch_inv]

            # queries
            batch_query_ids = query_ids[mask]
            batch_q = q[batch_query_ids]
            assert batch_q.shape == batch_k.shape, (batch_q.shape, batch_k.shape)

            # dist
            d = dist_func(batch_q, batch_k)
            assert d.shape == (batch_q.shape[0],), (d.shape, batch_q.shape)

            # update
            mask = mask.reshape(-1, k)
            tmp[mask] = d



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
    parser.add_argument('--skip-lookup', action='store_true')
    parser.add_argument('--batch-size', default=4096, type=int)
    args = parser.parse_args()

    # Print flags.
    print(args)

    os.system('mkdir -p {}'.format(args.output))

    main(args)

