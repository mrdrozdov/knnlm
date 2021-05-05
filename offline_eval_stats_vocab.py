"""
There are a few important sets of datastructures:

    dimensions

        * N - Size of the dstore.
        * K - Number of retrieved neighbors.
        * D - Size of the key vectors.

    dstore - This is the "ground truth" source of keys, values, and other important
        items created by the KNN-LM.

        * dstore_keys.npy - The vectors. NxD
        * dstore_vals.npy - The source token. Note: These are NOT the values used in the KNN-LM paper. Nx1
        * dstore_tgts.npy - The target token. Note: These ARE the values used in the KNN-LM paper. Nx1
        * dstore_prob.npy - The predicted probability of the target token. This can be used to compute perplexity of the non-retrieval model. Nx1

    lookup - This is a cache of retrieved neighbors on a subset of the dstore.

        * lookup_knns.npy - The retrieved neighbors. NxKx1
        * lookup_dist.npy - The approximate distance determined by product quantization and faiss. NxKx1
        * lookup_done.npy - We only compute `knns` and `dist` for a subset of the data, and we can use `done` to keep track
            of which rows we did this for. If `done` is 1, then the row has been computed, and 0 otherwise. Nx1
"""


import argparse
import collections
import json
import os
import sys

from vocab import Dictionary

import numpy as np
import torch

from tqdm import tqdm

_my_globals = {}


class DownsampleDemo:

    @staticmethod
    def print_statistics(context):
        args = context['args']
        knn_dstore = context['knn_dstore']
        vocab = context['vocab']
        tgts = context['tgts']
        knns = context['knns']
        knn_tgts = context['knn_tgts']
        u_knns, indices, knn_counts = context['u_knns']
        u_tgts = knn_tgts.reshape(-1)[indices]
        u_tgts_, knn_tgt_counts = context['u_knn_tgts']

        use_cuda = False
        device = torch.cuda.current_device() if use_cuda else None

        #
        vocab_size = len(vocab)
        data_size = knns.reshape(-1).shape[0]
        num_entries = knns.shape[0]
        num_unique_knns = u_tgts.shape[0]
        num_unique_tgts = u_tgts_.shape[0]

        # Print number of hits per token.
        if False:
            total = data_size
            with open('out/num_hits_per_token.out', 'w') as f:
                for i in tqdm(range(num_unique_tgts), desc='hits'):
                    idx = u_tgts_[i]
                    sym = vocab.symbols[idx]
                    count_ = knn_tgt_counts[i]
                    line = '{} idx={} {} {}/{} {:.3f}'.format(
                        i, idx, sym, count_, total, count_ / total)
                    f.write(line + '\n')

        # Print number of unique entries per token.
        if False:
            batch_size = 16
            pt_u_tgts = torch.from_numpy(u_tgts).to(device)
            with open('out/num_unique_keys_per_token.out', 'w') as f:
                for start in tqdm(range(0, num_unique_tgts, batch_size), desc='unique-entries'):
                    batch_i = np.arange(start, min(start + batch_size, num_unique_tgts))
                    batch_tgts = u_tgts_[batch_i]
                    batch_tgts = torch.from_numpy(batch_tgts).to(device)
                    mask = batch_tgts.view(-1, 1) == pt_u_tgts.view(1, -1)
                    batch_count_ = mask.sum(-1) # num unique keys per token.
                    #batch_count_ = np.sum(batch_idx.reshape(-1, 1) == u_tgts.reshape(1, -1), axis=-1)
                    #import ipdb; ipdb.set_trace()

                    for i_b, (i, idx) in enumerate(zip(batch_i, batch_tgts)):
                        sym = vocab.symbols[idx]
                        count_ = batch_count_[i_b]
                        line = '{} {} {}'.format(idx, sym, count_)
                        f.write(line + '\n')

                    del mask

        # Choose the keys to discard.
        _ = """
        If a token has more than `threshold` unique keys, then attempt to filter down.

        This should output:
            - List of positive knns (to keep).
            - And knns to discard.
        """
        batch_size = 4
        vocab_threshold = 1000
        top_N = 5000
        all_knns = np.arange(knn_dstore.tgts.shape[0])
        all_tgts = npy_copy(knn_dstore.tgts[:]).reshape(-1)
        is_active = set()
        to_keep = set()
        to_discard = set()
        #
        pt_u_tgts = torch.from_numpy(u_tgts).to(device)

        def run_batch(i):
            # TODO: This could be greatly sped up if you argsort u_knns first, according to u_tgts.

            # Get output vocab.
            start = i
            end = min(i + batch_size, vocab_threshold)
            #
            u_range = np.arange(0, u_tgts_.shape[0])
            u_tgts_start = u_range[u_tgts_ >= start].min()
            u_tgts_end = u_range[u_tgts_ < end].max() + 1
            batch_tgts = u_tgts_[u_tgts_start:u_tgts_end]
            del u_tgts_start, u_tgts_end
            batch_size_ = batch_tgts.shape[0]

            if batch_size_ == 0:
                return

            # This works because u_tgts (which are tgts associated with u_knns) is sorted by tgt.
            u_range = np.arange(0, u_tgts.shape[0])
            u_tgts_start = u_range[u_tgts >= batch_tgts[0].item()].min()
            u_tgts_end = u_range[u_tgts <= batch_tgts[-1].item()].max() + 1
            batch_knns = u_knns[u_tgts_start:u_tgts_end]
            batch_knn_tgts = u_tgts[u_tgts_start:u_tgts_end]
            batch_knn_counts = knn_counts[u_tgts_start:u_tgts_end].copy()

            assert batch_knns.shape == batch_knn_tgts.shape
            assert batch_knns.shape == batch_knn_counts.shape

            # Get mask.
            mask_is_tgt = batch_tgts.reshape(-1, 1) == batch_knn_tgts.reshape(1, -1)

            # Sort by value and slice to top.
            pt_mask_is_tgt = torch.from_numpy(mask_is_tgt).to(device)
            batch_knn_counts_repeat = torch.from_numpy(batch_knn_counts).to(device)
            batch_knn_counts_repeat = batch_knn_counts_repeat.view(1, -1).repeat(batch_size_, 1)
            batch_knn_counts_repeat[pt_mask_is_tgt == False] = 0
            index = torch.topk(batch_knn_counts_repeat, k=top_N, dim=1, largest=True, sorted=True).indices

            # Mask for top-N.
            mask_is_top = torch.BoolTensor(*mask_is_tgt.shape).to(device).fill_(False)
            mask_is_top.scatter_(index=index, dim=1, src=torch.ones(index.shape, dtype=torch.bool, device=device))
            mask_is_top = mask_is_top.cpu().numpy()

            # Everything marked as true should be kept.
            mask_keep = np.logical_and(mask_is_tgt, mask_is_top)
            mask_discard = np.logical_and(mask_is_tgt, mask_is_top == False)

            batch_knns_repeat = torch.from_numpy(batch_knns).view(1, -1).repeat(batch_size_, 1).numpy()

            to_keep.update(batch_knns_repeat[mask_keep].reshape(-1).tolist())
            to_discard.update(batch_knns_repeat[mask_discard].reshape(-1).tolist())
            is_active.update(batch_knns.reshape(-1).tolist())

        #
        for i in tqdm(range(0, vocab_threshold, batch_size), desc='filter'):
            run_batch(i)

        print('Filter Status')
        print('keep: {}'.format(len(to_keep)))
        print('discard: {}'.format(len(to_discard)))

        for knn, tgt in zip(all_knns.tolist(), all_tgts.tolist()):
            if knn in to_discard or knn in to_keep:
                continue
            if tgt >= vocab_threshold:
                to_keep.add(knn)
            else:
                to_discard.add(knn)

        print('Filter Status (+all)')
        print('keep: {}'.format(len(to_keep)))
        print('discard: {}'.format(len(to_discard)))

        with open(os.path.join(args.output, 'info.json'), 'w') as f:
            info = dict(keep=len(to_keep), discard=len(to_discard), active=len(is_active))
            f.write(json.dumps(info))

        # Write key ids.
        print('Write ids.')
        keep_ids = np.memmap(os.path.join(args.output, 'keep_ids.npy'), dtype=np.int, mode='w+', shape=(len(to_keep), 1))
        keep_ids[:, 0] = list(sorted(to_keep))
        disc_ids = np.memmap(os.path.join(args.output, 'discard_ids.npy'), dtype=np.int, mode='w+', shape=(len(to_discard), 1))
        disc_ids[:, 0] = list(sorted(to_discard))
        actv_ids = np.memmap(os.path.join(args.output, 'active_ids.npy'), dtype=np.int, mode='w+', shape=(len(is_active), 1))
        actv_ids[:, 0] = list(sorted(is_active))

        # Write key vectors.
        if args.write_keys:
            print('Write keys.')
            batch_size = 1024
            keys = knn_dstore.keys
            new_keys = np.memmap(os.path.join(args.output, 'dstore_keys.npy'), dtype=np.float32, mode='w+', shape=(len(to_keep), 1024))
            to_keep = list(sorted(to_keep))
            for start in tqdm(range(0, len(to_keep), batch_size), desc='write-keys'):
                end = min(start + batch_size, keys.shape[0])
                to_read = to_keep[start:end]
                new_keys[start:end] = keys[to_read]


    def run(self, context):
        vocab = context['vocab']
        knns = context['knns']
        knn_tgts = context['knn_tgts']

        vocab_size = len(vocab)
        data_size = knns.reshape(-1).shape[0]

        u, indices, knn_counts = np.unique(knns,
            return_index=True,
            return_counts=True
            )

        assert u.shape == indices.shape
        assert u.shape == knn_counts.shape

        # Sort according to tgts.
        u_tgts = knn_tgts.reshape(-1)[indices]
        sort_index = np.argsort(u_tgts)
        u = u[sort_index]
        indices = indices[sort_index]
        knn_counts = knn_counts[sort_index]

        # Unique tgts.
        u_tgts_, knn_tgt_counts = np.unique(knn_tgts, return_counts=True)

        # Run.
        context['u_knns'] = (u, indices, knn_counts)
        context['u_knn_tgts'] = (u_tgts_, knn_tgt_counts)
        DownsampleDemo.print_statistics(context)

        print('done')


def npy_copy(x):
    out = np.empty_like(x)
    out[:] = x
    return out


def main(args):
    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=True)

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)
    dstore.add_exact(args.lookup, args.lookup_k)
    #dstore.add_annotations(args.dstore)

    p = dstore.prob[:].copy()
    dist = -dstore.exact[:].copy()
    tgts = npy_copy(dstore.tgts[:])
    knn_tgts = npy_copy(dstore.knn_tgts[:, :args.k])
    knns = npy_copy(dstore.knns[:, :args.k])

    #
    limit = args.limit
    if limit > 0:
        p = p[:limit]
        dist = dist[:limit]
        tgts = tgts[:limit]
        knn_tgts = knn_tgts[:limit]
        knns = knns[:limit]

    #
    label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    #
    knn_p = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)
    flat_knn_logp, flat_knn_p = EvalUtil.get_knn_probmass(tgts, dist, knn_tgts)

    coeff = 0.25
    p_ = torch.from_numpy(p).float()
    knn_p_ = torch.from_numpy(knn_p).float()
    new_p = EvalUtil.combine_knn_and_vocab_probs(
                knn_p_,
                p_,
                coeff)
    ppl = EvalUtil.eval_ppl(p)
    new_ppl = EvalUtil.eval_ppl(new_p)

    print('ppl = {:.3f}, knn_ppl = {:.3f}'.format(ppl, new_ppl))

    #accum = np.zeros((vocab_size, 1), dtype=np.float32)

    #
    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))
    print('')

    # Demo.
    context = {}
    context['args'] = args
    context['vocab'] = vocab
    context['knns'] = knns
    context['tgts'] = tgts
    context['knn_tgts'] = knn_tgts
    context['knn_dstore'] = knn_dstore

    DownsampleDemo().run(context)


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
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
        # self.lookup_done = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))

    def add_exact(self, path, k):
        self.exact = np.memmap(os.path.join(path, 'lookup_exact.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))

    def add_annotations(self, path):
        self.src_pos = np.memmap(os.path.join(path, 'annotation_src_pos.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))

        # read pos dict
        self.idx2tag = []
        print('Reading POS Vocab...')
        path_pos_dict = os.path.join(path, 'pos_dict.txt')
        with open(path_pos_dict) as f:
            for line in f:
                print(line.strip())
                idx, sym, _ = line.strip().split()
                self.idx2tag.append(sym)
        self.tag2idx = {v: k for k, v in enumerate(self.idx2tag)}
        print('done\n')


class EvalUtil:
    @staticmethod
    def get_knn_probmass(tgts, dists, knn_tgts):
        tgts = torch.from_numpy(tgts).long().view(-1)
        dists = torch.from_numpy(dists).float().squeeze(-1)
        probs = torch.log_softmax(dists, dim=-1)
        mass = torch.exp(probs)
        return probs, mass

    @staticmethod
    def get_knn_log_prob(tgts, dists, knn_tgts):

        tgts = torch.from_numpy(tgts).long().view(-1)
        dists = torch.from_numpy(dists).float().squeeze(-1)
        #dists = -dists
        probs = torch.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(knn_tgts).long().squeeze(-1), tgts.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone().numpy()

        # Bx1
        return yhat_knn_prob.reshape(-1, 1)

    @staticmethod
    def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - coeff)
        coeffs[1] = np.log(coeff)
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

        return curr_prob

    @staticmethod
    def eval_ppl(p):
        avg_nll = -p.mean() / np.log(2)
        ppl = 2**avg_nll
        return ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    parser.add_argument('--vocab', default='data-bin/wikitext-103/dict.txt')
    # dstore neighbors
    parser.add_argument('--lookup', default='dstore_valid/lookup', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # dstore
    parser.add_argument('--knn-dstore', default='dstore_train', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    # output
    parser.add_argument('--output', default='filtered_dstore_train', type=str)
    parser.add_argument('--write-keys', action='store_true')
    # debug
    parser.add_argument('--limit', default=-1, type=int)
    args = parser.parse_args()

    # Print flags.
    print(args)

    # Write flags.
    os.system('mkdir -p {}'.format(args.output))
    with open(os.path.join(args.output, 'flags.json'), 'w') as f:
        f.write(json.dumps(args.__dict__))

    main(args)

