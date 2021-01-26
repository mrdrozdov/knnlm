
import argparse
import collections
import json
import os

from tqdm import tqdm

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', default='save_demo', type=str)
parser.add_argument('--k', default=16, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--demo', action='store_true')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


def load_data(load_dir):
    with open(os.path.join(load_dir, 'metadata.txt'), 'r') as f:
        metadata = json.loads(f.read())
    shape_lookup = {}
    for x in metadata['objects']:
        shape_lookup[x[0]] = tuple(x[1])
    data_size = None
    data = {}
    keys = ['target', 'queries', 'dists', 'knns', 'keys', 'vals', 'probs']
    for k in keys:
        shape = shape_lookup[k]
        v = np.memmap(os.path.join(load_dir, '{}.npy'.format(k)), mode='r', dtype=np.float32, shape=shape)
        print('load {} with shape {}'.format(k, v.shape))
        data[k] = v

        if data_size is None:
            data_size = v.shape[0]
        assert data_size == v.shape[0]

    return data, data_size


def compute_knn_probs(batch):
    target = batch['target']
    dists = batch['dists']
    vals = batch['vals']
    logprobs = torch.log_softmax(dists, dim=-1)
    index_mask = torch.eq(vals.long(), target.long().unsqueeze(-1)).float()
    index_mask[index_mask == 0] = -10000
    index_mask[index_mask == 1] = 0
    knn_logprobs = torch.logsumexp(logprobs + index_mask, dim=-1)
    return knn_logprobs


def batchify(data, offset, batch_size, cuda):
    batch = {}
    for k, v in data.items():
        x = v[offset:offset+batch_size]
        x = torch.from_numpy(x)
        if cuda:
            x = x.cuda()
        batch[k] = x
    return batch


def batch_limit(batch, limit):
    new_batch = {}
    for k, v in batch.items():
        new_batch[k] = v
    keys = ['dists', 'knns', 'keys', 'vals']
    for k in keys:
        new_batch[k] = batch[k][:, :limit]
    return new_batch


def find_optimal_batch(batch):
    new_batch = {}
    for k, v in batch.items():
        new_batch[k] = v
    keys = ['dists', 'knns', 'keys', 'vals']
    for k in keys:
        new_batch[k] = batch[k].clone()

    batch_size, limit = batch['vals'].shape

    # positives
    pos_offset = {i: 0 for i in range(batch_size)}
    for i in range(batch_size):
        dists = batch['dists'][i]
        vals = batch['vals'][i]
        tgt = batch['target'][i]
        mask = vals == tgt

        if not torch.any(mask):
            continue

        dists = dists[mask]
        vals = vals[mask]

        argsort = torch.argsort(dists, dim=0, descending=True)

        size = vals.shape[0]

        new_batch['dists'][i, :size] = dists[argsort]
        new_batch['vals'][i, :size] = vals[argsort]

        pos_offset[i] += size

    # negatives
    for i in range(batch_size):
        dists = batch['dists'][i]
        vals = batch['vals'][i]
        tgt = batch['target'][i]
        mask = vals != tgt

        if not torch.any(mask):
            continue

        dists = dists[mask]
        vals = vals[mask]

        argsort = torch.argsort(dists, dim=0, descending=False)

        offset = pos_offset[i]

        new_batch['dists'][i, offset:] = dists[argsort]
        new_batch['vals'][i, offset:] = vals[argsort]

    return new_batch


def find_optimal_batch_v2(batch):
    new_batch = {}
    for k, v in batch.items():
        new_batch[k] = v
    keys = ['dists', 'knns', 'keys', 'vals']
    for k in keys:
        new_batch[k] = batch[k].clone()

    batch_size, limit = batch['vals'].shape

    # positives
    pos_offset = {i: 0 for i in range(batch_size)}
    for i in range(batch_size):
        dists = batch['dists'][i]
        vals = batch['vals'][i]
        tgt = batch['target'][i]
        mask = vals == tgt

        if not torch.any(mask):
            continue

        dists = dists[mask]
        vals = vals[mask]

        argsort = torch.argsort(dists, dim=0, descending=True)

        size = vals.shape[0]

        new_batch['dists'][i, :size] = dists[argsort]
        new_batch['vals'][i, :size] = vals[argsort]

        pos_offset[i] += size

    # negatives
    for i in range(batch_size):
        dists = batch['dists'][i]
        vals = batch['vals'][i]
        tgt = batch['target'][i]
        mask = vals != tgt

        if not torch.any(mask):
            continue

        dists = dists[mask]
        vals = vals[mask]

        offset = pos_offset[i]

        new_batch['dists'][i, offset:] = dists
        new_batch['vals'][i, offset:] = vals

    return new_batch


def main(args):
    data, data_size = load_data(args.load_dir)

    batch_size = args.batch_size
    nbatches = data_size // batch_size
    if args.demo:
        nbatches = 5

    k_range = [2, 4, 8, 16, 32, 64]
    l_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results_at_k = {}
    for limit in k_range:
        results_at_k[limit] = collections.defaultdict(list)
        for l in l_range:
            results_at_k[(limit, l)] = collections.defaultdict(list)

    # compute probs
    offset = 0
    for i in tqdm(range(nbatches)):
        batch = batchify(data, offset, batch_size, args.cuda)

        target = batch['target']
        dists = batch['dists']
        vals = batch['vals']
        probs = torch.exp(batch['probs'])

        # use distance
        for limit in k_range:
            b = batch_limit(batch, limit)
            new_probs = torch.exp(compute_knn_probs(b))
            results_at_k[limit]['knn_probs'].append(new_probs.cpu().detach())

            for l in l_range:
                mix_probs = (1 - l) * probs + l * new_probs
                results_at_k[(limit, l)]['mix_probs'].append(mix_probs.cpu().detach())

        # find optimal batch
        obatch = find_optimal_batch(batch)
        for limit in k_range:
            b = batch_limit(obatch, limit)
            new_probs = torch.exp(compute_knn_probs(b))
            results_at_k[limit]['o_probs'].append(new_probs.cpu().detach())

            for l in l_range:
                mix_probs = (1 - l) * probs + l * new_probs
                results_at_k[(limit, l)]['mix_o_probs'].append(mix_probs.cpu().detach())

        # find optimal batch (v2)
        obatch = find_optimal_batch_v2(batch)
        for limit in k_range:
            b = batch_limit(obatch, limit)
            new_probs = torch.exp(compute_knn_probs(b))
            results_at_k[limit]['o_v2_probs'].append(new_probs.cpu().detach())

            for l in l_range:
                mix_probs = (1 - l) * probs + l * new_probs
                results_at_k[(limit, l)]['mix_o_v2_probs'].append(mix_probs.cpu().detach())

        # end
        offset += batch_size

    # print results
    print('knn')
    for limit in k_range:
        r = results_at_k[limit]
        probs = torch.cat(r['knn_probs'], 0)
        print(limit, probs.mean())
    print('optimal')
    for limit in k_range:
        r = results_at_k[limit]
        probs = torch.cat(r['o_probs'], 0)
        print(limit, probs.mean())
    print('optimal v2')
    for limit in k_range:
        r = results_at_k[limit]
        probs = torch.cat(r['o_v2_probs'], 0)
        print(limit, probs.mean())

    print('---')

    for key in ['mix_probs', 'mix_o_probs', 'mix_o_v2_probs']:
        for limit in k_range:
            for l in l_range:
                probs = torch.cat(results_at_k[(limit, l)][key], 0)
                out = probs.mean().item()
                print('("{}", {}, {}, {:.4f}),'.format(key, limit, l, out))




if __name__ == '__main__':
    main(args)

