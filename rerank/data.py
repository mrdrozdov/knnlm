"""
Two approaches:

    1. Nothing special. Read from memmap each batch.
    2. Every N batches read a subset of data into memory, and always read from this in-memory cache.


Suggested approach:

    1. At beginning of training sample N rows from the validation set to use for training
        where N should roughly fill memory.
    2. Run M experiments, and ensemble the approaches.


"""

import ctypes
import hurry.filesize
import mmap

import copy
import os
import time

import multiprocessing

import numpy as np
import torch

from tqdm import tqdm


class KNNDataset(torch.utils.data.Dataset):
    def __init__(self, dstore, knn_dstore, mp=False):
        super().__init__()
        self.dstore = dstore
        self.knn_dstore = knn_dstore
        self.in_memory = InMemoryDataset(self, mp=mp)

    def __len__(self):
        return self.dstore.tgts.shape[0]

    def __getitem__(self, index):
        item = self.dstore.tgts[index]
        return index, item


class KNNFoldDataset(torch.utils.data.Dataset):
    def __init__(self, fold, fold_info):
        super().__init__()
        self.fold = fold
        self.fold_info = fold_info

    def __len__(self):
        return self.fold_info.index.shape[0]

    def __getitem__(self, index):
        item = self.fold_info.index[index]
        return index, item


class Worker(multiprocessing.Process):
    def __init__(self, input_dict, result_dict):
        super().__init__()
        self.input_dict = input_dict
        self.result_dict = result_dict

    def run(self, *args, **kwargs):
        input_dict, result_dict = self.input_dict, self.result_dict
        result_dict['u_keys'] = KeysUtil.fancy_read(**input_dict)


class KeysUtil:
    @staticmethod
    def fancy_read(keys=None, path=None, offset=None, shape=None, u=None, batch_size=500000, position=None):
        if keys is None:
            # Open file.
            keys = np.memmap(path, dtype=np.float32, mode='r', shape=shape, offset=offset)

        n = keys.shape[0]
        u_keys = np.empty((u.shape[0], 1024), dtype=np.float32)
        u_range = np.arange(u.shape[0])
        u_offset = 0
        for start in tqdm(range(0, n, batch_size), desc='fancy_read', position=position):
            end = min(start + batch_size, n)

            # early exit
            if u_offset >= u.shape[0]:
                break

            # select u start
            u_start = u_offset
            if start > u[u_start]:
                continue
            if u[u_start] >= end:
                continue
            assert u[u_start] >= start
            assert u[u_start] < end, (u[u_start], start, end)

            # select u end
            u_end = u_start
            for i in range(u_start, u.shape[0]):
                if u[i] >= end:
                    break
                u_end = i
            u_offset = u_end + 1
            assert u[u_end] < end

            # keys

            # Read chunk.
            if True:
                batch_u = u[u_start:u_end+1]
                chunk_k = keys[start:end]
                batch_k = chunk_k[batch_u - start]

            # Flexible read.
            if False:
                batch_u = u[u_start:u_end+1]
                batch_k = keys[batch_u]

            u_keys[u_start:u_end+1] = batch_k

            # madvise
            madvise = ctypes.CDLL("libc.so.6").madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int

            assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

        return u_keys

    @staticmethod
    def offset_func(offset, rowsize=1024):
        return offset * rowsize * int(32 / 8)


class InMemoryDataset:
    def __init__(self, dataset, max_size=4 * (10**6), batch_size=32, mp=False):
        self.dataset = dataset
        self.max_size = max_size
        self.batch_size = batch_size
        self.mp = mp
        self.reset()

    def reset(self):
        max_size = self.max_size
        batch_size = self.batch_size
        shape = (max_size + batch_size, 1024)
        self.u_keys = None
        self.u_knns_ = set()
        self.batches = []
        self.ready = False

    def update(self, batch):
        self.batches.append(batch)

        knns = self.dataset.dstore.knns[batch]
        u = np.unique(knns)
        self.u_knns_.update(u.tolist())

        if len(self.u_knns_) >= self.max_size:
            self.load()

    def fancy_read(self, u):
        if self.mp:
            return self.mp_fancy_read(u)

        knn_dstore = self.dataset.knn_dstore
        keys = knn_dstore.keys

        if True:
            # https://github.com/numpy/numpy/issues/13172#issue-423761095
            madvise = ctypes.CDLL("libc.so.6").madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int

            assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

        u_keys = KeysUtil.fancy_read(keys=keys, u=u)
        return u_keys

    def mp_fancy_read(self, u):
        knn_dstore = self.dataset.knn_dstore
        path = os.path.join(knn_dstore.path, 'dstore_keys.npy')
        N = knn_dstore.keys.shape[0]

        u_keys = np.empty((u.shape[0], 1024), dtype=np.float32)

        num_workers = 2
        jobs = []
        results = []
        masks = []

        for i in range(num_workers):
            chunk_size = N // num_workers
            start = i * chunk_size
            if i < num_workers - 1:
                end = start + chunk_size
            else:
                end = N

            offset = KeysUtil.offset_func(start)
            shape = (end - start, 1024)

            mask = np.logical_and(u >= start, u < end)
            batch_u = u[mask]
            masks.append(mask)

            input_dict = {}
            input_dict['path'] = path
            input_dict['offset'] = offset
            input_dict['shape'] = shape
            input_dict['u'] = batch_u - start
            input_dict['position'] = i

            result_dict = {}
            results.append(result_dict)

            print(i, batch_u.shape)

            p = Worker(input_dict=input_dict, result_dict=result_dict)
            jobs.append(p)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        for r, m in zip(results, masks):
            batch_u_keys = r['u_keys']
            u_keys[m] = batch_u_keys

        return u_keys

    def load(self):
        batch = np.concatenate(self.batches)
        knns = self.dataset.dstore.knns[batch]
        u, inverse = np.unique(knns, return_inverse=True)
        assert len(u) == len(self.u_knns_)
        print('fancy load for knns = {} with unique = {}'.format(knns.shape, u.shape))
        u_keys = self.fancy_read(u)
        self.u_knns = u
        self.u_knns_inverse = inverse
        self.u_keys = u_keys
        self.ready = True

    def get_batches(self, force=False):
        if force and not self.ready:
            self.load()
        for b in self.batches:
            yield b
        self.reset()


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=32, include_partial=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.include_partial = include_partial

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        batch_size = self.batch_size
        n = len(self.dataset)
        order = np.arange(n)
        np.random.shuffle(order)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = order[start:end]
            if not self.include_partial and len(batch) < batch_size:
                break
            yield batch


def build_collate_fold(context, dataset):
    def custom_collate_fn(batch):
        index, items = zip(*batch)
        index = np.array(index)
        items = np.array(items)

        # vars
        batch_size = items.shape[0]
        fold = dataset.fold
        fold_info = dataset.fold_info
        knns = context['dstore'].knns
        knn_tgts = context['dstore'].knn_tgts
        dist = context['dstore'].dist
        tgts = context['dstore'].tgts
        knns_mask = fold_info.mask
        k = knns.shape[1]
        assert knns.shape == knns_mask.shape

        # batch
        batch_knns = knns[items]
        batch_knn_tgts = knn_tgts[items]
        batch_dist = dist[items]
        batch_tgts = tgts[items]
        batch_mask = knns_mask[items]

        # batch keys
        batch_keys = np.zeros((batch_size * k, 1024), dtype=np.float32)
        knns_ = batch_knns[batch_mask]
        idx_ = np.array([fold_info.knn_TO_idx[x] for x in knns_.tolist()])
        batch_keys[batch_mask.reshape(-1)] = fold_info.u_keys[idx_]
        batch_keys = batch_keys.reshape(batch_size, k, 1024)

        # batch_map
        batch_map = {}
        batch_map['mask'] = batch_mask
        batch_map['keys'] = batch_keys
        batch_map['knns'] = batch_knns
        batch_map['knn_tgts'] = batch_knn_tgts
        batch_map['dist'] = batch_dist
        batch_map['tgts'] = batch_tgts

        return batch_map
    return custom_collate_fn


def build_collate(dataset):
    def custom_collate_fn(batch):
        index, items_list = zip(*batch)

        print('[collate] PID = {}, first = {}'.format(os.getpid(), index[0]))

        index = np.array(index)

        batch_map = {}
        batch_map['index'] = index
        batch_map['items_list'] = items_list

        # TODO: Use the in-memory dataset.
        if False:
            dstore = dataset.dstore
            knn_dstore = dataset.knn_dstore

            knns = dstore.knns[index]

            print('uniq')
            u, inverse = np.unique(knns, return_inverse=True)
            u_keys = np.zeros((u.shape[0], 1024), dtype=np.float32)

            def offset_fn(offset, rowsize=1024):
                return offset * rowsize * int(32 / 8)

            print('read')
            u_keys = knn_dstore.keys[u]

            print('write')
            keys = u_keys[inverse]
            batch_map['keys'] = keys

        return batch_map
    return custom_collate_fn


class Dstore:
    def __init__(self, path, dstore_size, vec_size=1024):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self, include_keys=False):
        path = self.path
        if include_keys:
            self.keys = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.tgts = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.prob = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        self._initialized = True

    def add_neighbors(self, path, k):
        self.knns = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.knn_tgts = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))

    def add_exact(self, path, k):
        self.exact = np.memmap(os.path.join(path, 'lookup_exact.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))


class InMemoryDstore:
    def __init__(self, dstore):
        self.dstore = dstore

    @staticmethod
    def from_dstore(dstore, keys=['tgts', 'knns', 'knn_tgts', 'exact']):
        new_dstore = InMemoryDstore(dstore)
        for k in keys:
            if k == 'exact':
                x = -npy_copy(getattr(dstore, k))
                setattr(new_dstore, 'dist', x)
            else:
                x = npy_copy(getattr(dstore, k))
                setattr(new_dstore, k, x)
        return new_dstore


def npy_copy(x):
    out = np.empty(x.shape, dtype=x.dtype)
    out[:] = x
    return out


class FoldInfo:
    def __init__(self, fold, index, mask, u_knns, u_keys):
        self.fold = fold
        self.index = index
        self.mask = mask
        self.u_knns = u_knns
        self.u_keys = u_keys


def select_fold_knns_and_keys(context, fold, include_first=16, max_keys=1000000, max_rows=20000):

    dstore = context['dstore']
    knn_dstore = context['knn_dstore']
    knns = dstore.knns
    n = knns.shape[0]

    print('[fold] shape={}'.format(fold.shape))

    index = fold.copy()
    np.random.shuffle(index)
    index = index[:max_rows]

    u_first = np.unique(knns[index, :include_first])

    print('[fold] include_first = {}, u_first = {}'.format(include_first, u_first.shape[0]))

    u_all = set(u_first.tolist())

    knn_mask = np.zeros(knns.shape, dtype=np.bool)
    knn_mask[index, :include_first] = True

    bucket_size = 128
    num_buckets = 1024 // bucket_size

    max_iterations = 100
    for i in range(max_iterations):
        local_knn = knns[index]
        local_mask = np.random.choice([True, False], size=local_knn.shape, p=[0.025, 0.975])
        local_mask = np.logical_and(knn_mask[index] == False, local_mask)
        local_knns_ = local_knn[local_mask]
        u_local = np.unique(local_knns_)

        # Update count.
        u_all.update(u_local.tolist())

        # Update mask.
        knn_mask[index] = np.logical_or(knn_mask[index], local_mask)

        print('i = {}, u_all = {}, size = {}'.format(i, len(u_all), hurry.filesize.size(len(u_all) * 1024 * 4)))

        if len(u_all) > max_keys:
            break

    u_knns = np.array(sorted(u_all))
    u_keys = KeysUtil.fancy_read(keys=knn_dstore.keys, u=u_knns)
    del u_all

    knn_TO_idx = {x: i for i, x in enumerate(u_knns.tolist())}

    fold_info = FoldInfo(fold, index, mask=knn_mask, u_knns=u_knns, u_keys=u_keys)
    fold_info.knn_TO_idx = knn_TO_idx

    return fold_info


def build_fold_for_epoch(context, total=10, fold_id=0, max_keys=1000000, include_first=16, max_rows=20000):
    """
    1. Randomly select max rows from training. Only choose from the rows valid for training.
    2. From selected, choose up a number of knns without exceeding max keys.
    3. The remaining fold is used for validation.
    """

    dstore = context['dstore']
    knn_dstore = context['knn_dstore']
    knns = dstore.knns
    n = knns.shape[0]

    start = (fold_id // total) * n
    end = min(start + (n // total), n)
    dev_fold = np.arange(start, end)

    if fold_id == 0:
        start = dev_fold[-1] + 1
        end = n
        trn_fold = np.arange(start, end)
    elif fold_id == total - 1:
        start = 0
        end = dev_fold[0]
        trn_fold = np.arange(start, end)
    else:
        assert fold_id < total and fold_id > 0
        start = 0
        end = dev_fold[0]
        trn_fold = np.arange(start, end)
        start = dev_fold[-1] + 1
        end = n
        trn_fold = np.concatenate(trn_fold, np.arange(start, end))

    trn_fold_info = select_fold_knns_and_keys(context, trn_fold, include_first=include_first, max_keys=max_keys, max_rows=max_rows)

    context['trn_fold'] = trn_fold
    context['trn_fold_info'] = trn_fold_info

    return context


def demo():
    num_workers = args.n_workers
    batch_size = args.batch_size

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)
    dstore.add_exact(args.lookup, args.lookup_k)
    dstore_ = InMemoryDstore.from_dstore(dstore)

    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=True)

    # build fold
    context = {}
    context['dstore'] = dstore_
    context['knn_dstore'] = knn_dstore

    if args.demo:
        context = build_fold_for_epoch(context, total=2, fold_id=0, max_keys=100000, max_rows=1000)

    else:
        context = build_fold_for_epoch(context, total=10, fold_id=0, max_keys=1000000)

    # build dataset, sampler, loader
    trn_fold = context['trn_fold']
    trn_fold_info = context['trn_fold_info']
    dataset = KNNFoldDataset(trn_fold, trn_fold_info)
    sampler = BatchSampler(dataset, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(sampler is None),
        num_workers=num_workers,
        batch_sampler=sampler,
        collate_fn=build_collate_fold(context, dataset),
        )

    for batch_map in loader:
        mask = batch_map['mask']
        rmin = mask.sum(1).min()
        rmax = mask.sum(1).max()
        ravg = mask.sum(1).mean()
        print('[batch] row(min = {}, max = {}, mean = {}'.format(rmin, rmax, ravg))
        time.sleep(1)


    sys.exit()
    #

    dataset = KNNDataset(dstore_, knn_dstore, mp=args.mp)
    sampler = BatchSampler(dataset, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(sampler is None),
        num_workers=num_workers,
        batch_sampler=sampler,
        collate_fn=build_collate(dataset),
        )

    for batch_map in loader:
        index = batch_map['index']
        print('[batch] PID = {}, first = {}'.format(os.getpid(), index[0]))
        time.sleep(1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    #
    parser.add_argument('--knn-dstore', default='dstore_train', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    # dstore neighbors
    parser.add_argument('--lookup', default='dstore_valid/lookup', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # dstore
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--n-workers', default=0, type=int)
    parser.add_argument('--mp', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    demo()

