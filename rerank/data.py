"""
Two approaches:

    1. Nothing special. Read from memmap each batch.
    2. Every N batches read a subset of data into memory, and always read from this in-memory cache.


"""


import os
import time

import numpy as np
import torch

from tqdm import tqdm


class KNNDataset(torch.utils.data.Dataset):
    def __init__(self, dstore, knn_dstore):
        super().__init__()
        self.dstore = dstore
        self.knn_dstore = knn_dstore
        self.in_memory = InMemoryDataset(self)

    def __len__(self):
        return self.dstore.tgts.shape[0]

    def __getitem__(self, index):
        item = self.dstore.tgts[index]
        return index, item


class InMemoryDataset:
    def __init__(self, dataset, max_size=2 * (10**6), batch_size=32):
        self.dataset = dataset
        self.max_size = max_size
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        max_size = self.max_size
        batch_size = self.batch_size
        shape = (max_size + batch_size, 1024)
        self.keys = np.empty(shape, dtype=np.float32)
        self.u_knns = set()
        self.batches = []
        self.ready = False

    def update(self, batch):
        self.batches.append(batch)

        knns = self.dataset.dstore.knns[batch]
        u = np.unique(knns)
        self.u_knns.update(u.tolist())

        if len(self.u_knns) >= self.max_size:
            self.load()

    def load(self):
        for knn in tqdm(self.u_knns, desc='load'):
            # TODO: Fancy read.
            pass
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

        d = self.dataset.in_memory

        def i_():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = order[start:end]
                if not self.include_partial and len(batch) < batch_size:
                    break
                yield batch
            yield None

        for batch in i_():
            done = batch is None
            if not done:
                d.update(batch)

            if d.ready or done:
                for b in d.get_batches(force=done):
                    yield b


def build_collate(dataset):
    def custom_collate_fn(batch):
        index, items_list = zip(*batch)

        print('[collate] PID = {}, first = {}'.format(os.getpid(), index[0]))

        index = np.array(index)

        batch_map = {}
        batch_map['index'] = index
        batch_map['items_list'] = items_list

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


class InMemoryDstore:
    def __init__(self, dstore):
        self.dstore = dstore

    @staticmethod
    def from_dstore(dstore, keys=['tgts', 'knns', 'knn_tgts']):
        new_dstore = InMemoryDstore(dstore)
        for k in keys:
            x = npy_copy(getattr(dstore, k))
            setattr(new_dstore, k, x)
        return new_dstore


def npy_copy(x):
    out = np.empty(x.shape, dtype=x.dtype)
    out[:] = x
    return out


def demo():
    num_workers = args.n_workers
    batch_size = args.batch_size

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)
    dstore_ = InMemoryDstore.from_dstore(dstore)

    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=True)

    dataset = KNNDataset(dstore_, knn_dstore)
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
    args = parser.parse_args()
    demo()

