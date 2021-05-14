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
            nn.Linear(2 * input_size, size),
            nn.ReLU(),
            nn.Linear(size, 1),
            )

    def forward(self, q, x):
        inp = torch.cat([q, x], 1)
        return self.enc(inp)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, b, s, y):

        r = torch.arange(b.shape[0], device=b.device)

        m_b = b.view(-1, 1) == b.view(1, -1) # same batch id
        m_y = torch.logical_xor(y.view(-1, 1), y.view(1, -1)) # different labels
        m_r = r.view(-1, 1) < r.view(1, -1) # prevent duplicates
        m = torch.logical_and(torch.logical_and(m_b, m_y), m_r)

        if m.sum().item() == 0:
            raise EmptyBatchException

        mat_d = s.view(-1, 1) - s.view(1, -1)
        mat_y = y.view(-1, 1).repeat(1, y.shape[0])
        d = mat_d[m]
        z = mat_y[m].float()

        loss = nn.BCEWithLogitsLoss()(d, z)

        output = {}
        output['loss'] = loss
        output['logits'] = d.detach()
        output['labels'] = z.detach()

        return output


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

        # TODO: Move this into collate?
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

        #b, x, y = self.clean_batch(tgts, knn_tgts, keys, mask)
        device = self.device
        b = batch_map['b'].to(device)
        q = batch_map['q'].to(device)
        x = batch_map['x'].to(device)
        y = batch_map['y'].to(device)

        s = self.enc(q, x)

        loss_output = self.loss(b, s, y)

        loss = loss_output['loss']

        output = {}
        output['loss'] = loss
        output['logits'] = loss_output['logits']
        output['labels'] = loss_output['labels']

        return output


def run_eval(batch_map, model_output):
    logits = model_output['logits']
    labels = model_output['labels']

    pred = torch.sigmoid(logits).round()
    assert torch.all(pred >= 0).item() == True
    assert torch.all(pred <= 1).item() == True

    correct = (pred == labels).sum().item()
    total = labels.shape[0]

    output = {}
    output['correct'] = correct
    output['total'] = total

    return output




def main(args):
    num_workers = args.n_workers
    batch_size = args.batch_size

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize(include_keys=True)
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
        context = build_fold_for_epoch(context, total=2, fold_id=0, include_first=8, max_keys=1000, max_rows=100, skip_read=args.skip_read)

    else:
        context = build_fold_for_epoch(context, total=10, fold_id=0, max_keys=1000000, skip_read=args.skip_read)

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
                model_output = net(batch_map)
            except EmptyBatchException:
                epoch_debug['skip'] += 1
                continue

            loss = model_output['loss']
            opt.zero_grad()
            loss.backward()
            opt.step()

            eval_output = run_eval(batch_map, model_output)

            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['correct'].append(eval_output['correct'])
            epoch_metrics['total'].append(eval_output['total'])

        print('epoch: {}'.format(epoch))
        print('epoch-debug: {}'.format(json.dumps(epoch_debug)))
        print('avg-loss: {:.3f}'.format(np.mean(epoch_metrics['loss'])))

        correct = np.sum(epoch_metrics['correct'])
        total = np.sum(epoch_metrics['total'])
        acc = correct / total

        print('acc: {:.5f} ({}/{})'.format(acc, correct, total))

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
    parser.add_argument('--skip-read', action='store_true')
    args = parser.parse_args()
    main(args)

