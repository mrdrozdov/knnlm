import collections
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from data import *


class EmptyBatchException(Exception):
    pass


def pick(d, keys):
    return [d[k] for k in keys]


class Encoder(nn.Module):
    def __init__(self, input_size=1024, size=100):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_size, size),
            nn.ReLU(),
            nn.Linear(size, 1),
            )

    def forward(self, x):
        return self.enc(x)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, b, s, y):
        new_b = []
        d = []
        z = []

        u_b = torch.unique(b)

        for i_b in u_b.tolist():
            mask = b == i_b
            m = mask.sum().item()

            local_s = s[mask]
            local_y = y[mask]

            for i in range(m):
                for j in range(m):
                    if j <= i:
                        continue
                    if local_y[i].item() == local_y[j].item():
                        continue

                    new_b.append(i_b)
                    d.append(local_s[i] - local_s[j])
                    z.append(local_y[i].item() == True)

        if len(d) == 0:
            raise EmptyBatchException

        device = y.device
        new_b = torch.tensor(new_b, dtype=torch.long, device=device)
        d = torch.cat(d)
        z = torch.tensor(z, dtype=torch.float, device=device)

        loss = nn.BCEWithLogitsLoss()(d, z)

        return loss


class Net(nn.Module):
    @staticmethod
    def from_context(context):
        args = context['args']
        enc = Encoder()
        loss = Loss()
        net = Net(enc, loss)
        return net

    def __init__(self, enc, loss):
        super().__init__()
        self.enc = enc
        self.loss = loss

    @property
    def device(self):
        return next(self.parameters()).device

    def clean_batch(self, tgts, knn_tgts, keys, mask):
        """
        Returns:

            b: The batch index.
            x: The key vector.
            y: True if knn_tgts == tgts.
        """
        device = self.device
        m = mask.sum()
        batch_size = tgts.shape[0]
        k = 1024
        input_size = 1024

        b = torch.zeros(m, dtype=torch.long, device=device)
        x = torch.zeros(m, input_size, dtype=torch.float, device=device)
        y = torch.zeros(m, dtype=torch.long, device=device)

        b[:] = torch.from_numpy(np.arange(batch_size).repeat(k).reshape(batch_size, k)[mask.reshape(batch_size, k)]).long().to(device)
        x[:] = torch.from_numpy(keys.reshape(-1, input_size)[mask.reshape(-1)]).float().to(device)

        batch_tgts = tgts[np.arange(batch_size).repeat(k).reshape(batch_size, k)[mask.reshape(batch_size, k)]].reshape(-1)
        batch_knn_tgts = knn_tgts[mask].reshape(-1)

        y[:] = torch.from_numpy(batch_tgts == batch_knn_tgts).bool().to(device)

        return b, x, y

    def forward(self, batch_map):
        knns, knn_tgts, dist = pick(batch_map, ['knns', 'knn_tgts', 'dist'])
        keys = batch_map['keys']
        mask = batch_map['mask']
        tgts = batch_map['tgts']

        b, x, y = self.clean_batch(tgts, knn_tgts, keys, mask)

        s = self.enc(x)
        loss = self.loss(b, s, y)

        return loss



def main(args):
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
    context['args'] = args
    context['dstore'] = dstore_
    context['knn_dstore'] = knn_dstore

    if args.demo:
        context = build_fold_for_epoch(context, total=2, fold_id=0, include_first=8, max_keys=1000, max_rows=100)

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

    # net
    net = Net.from_context(context)
    if args.cuda:
        net.cuda()
    opt = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.max_epoch):

        epoch_debug = collections.Counter()
        epoch_metrics = collections.defaultdict(list)

        for batch_map in tqdm(loader, desc='train'):
            try:
                loss = net(batch_map)
            except EmptyBatchException:
                epoch_stats['skip'] += 1
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_metrics['loss'].append(loss.item())

        print('epoch: {}'.format(epoch))
        print('epoch-debug: {}'.format(json.dumps(epoch_debug)))
        print('avg-loss: {:.3f}'.format(np.mean(epoch_metrics['loss'])))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    parser.add_argument('--lookup', default='dstore_valid/lookup', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # knn dstore
    parser.add_argument('--knn-dstore', default='dstore_train', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    # training
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--max-epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    # debug
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n-workers', default=0, type=int)
    parser.add_argument('--mp', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    main(args)

