import torch
import faiss
import math
import numpy as np
import time

from dstore_util import Dstore
from dataset_util import *


class KNN_Index:
    def __init__(self, path, nprobe, k=1024):
        index = faiss.read_index(path)
        index.nprobe = nprobe
        self.index = index
        self.k = k

    def get_knns(self, queries):
        dists, knns = self.index.search(queries, self.k)
        return dists, knns


def demo():
    path = 'filtered_dstore_train:keep_non_active=false/info.json'
    with open(path) as f:
        info = json.loads(f.read())
    print(json.dumps(info))
    path = 'filtered_dstore_train:keep_non_active=false/keep_ids.npy'
    keep_ids = np.memmap(path, dtype=np.int, mode='r', shape=(info['keep'], 1))
    keep_ids = npy_copy(keep_ids[:]).reshape(-1)

    print('small index')
    k = 1024
    nprobe = 32
    path = 'filtered_dstore_train:keep_non_active=false/knn.index'
    small_index = KNN_Index(path, nprobe, k)

    print('big index')
    k = 1024
    nprobe = 32
    path = 'dstore_train/knn.index'
    big_index = KNN_Index(path, nprobe, k)

    print('dstore')
    dstore = Dstore('dstore_valid', 217646, 1024)
    dstore.initialize(include_keys=True)

    p = npy_copy(dstore.prob)
    q = npy_copy(dstore.keys)
    tgts = npy_copy(dstore.tgts)

    limit = 1000
    if limit > 0:
        p = p[:limit]
        q = q[:limit]
        tgts = tgts[:limit]

    print('query')
    res = {}
    res['sm'] = small_index.get_knns(q)
    res['bg'] = big_index.get_knns(q)



    import ipdb; ipdb.set_trace()
    pass


if __name__ == '__main__':

    demo()

