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


class RunOriginal:
    def run(self, dstore, vocab, mask=None, mask_b=None, tag=None):
        if mask is None:
            p = dstore.prob[:].copy()
        else:
            p = dstore.prob[:].copy()[mask]
        p_ = torch.from_numpy(p).float()
        ppl = EvalUtil.eval_ppl(p_)
        out = {}
        out['ppl'] = ppl
        out['cfg'] = str(None)
        out['desc'] = 'original[shape={}]'.format(p.shape[0])
        return out


class RunKNNLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_exact = False
        self.flip_distance = True
        self.sort = True
        for k, v in cfg.items():
            setattr(self, k, v)

    def run(self, dstore, vocab, mask=None, mask_b=None, tag=None):
        if mask is None:
            p = dstore.prob[:].copy()
            dist = dstore.dist[:].copy()
            knn_tgts = dstore.knn_tgts[:].copy()
            tgts = dstore.tgts[:].copy()
        else:
            p = dstore.prob[:].copy()[mask]
            dist = dstore.dist[:].copy()[mask]
            knn_tgts = dstore.knn_tgts[:].copy()[mask]
            tgts = dstore.tgts[:].copy()[mask]

        if self.use_exact:
            dist = dstore.exact[:].copy()
            if mask is not None:
                dist = dist[mask]

        if self.flip_distance:
            dist = -dist

        index = None
        if self.sort:
            assert len(dist.shape) == 3
            index = np.argsort(dist, axis=1)[:, ::-1]
            dist = np.take_along_axis(dist, index, axis=1)
            knn_tgts = np.take_along_axis(knn_tgts, index, axis=1)

        p_ = torch.from_numpy(p).float()
        original_ppl = EvalUtil.eval_ppl(p_)

        best_val = None
        best_knn_p = None
        best_cfg = None
        limits_to_check = 8
        limit_size = self.k // limits_to_check
        limits_to_check_lst = [i * limit_size for i in range(1, limits_to_check)]
        if not self.find_best_lim:
            limits_to_check_lst = [self.k]
        coeff_lst = np.arange(20) / 20


        for lim in limits_to_check_lst:
            dist_ = dist[:, :lim]
            knn_tgts_ = knn_tgts[:, :lim]
            knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)
            knn_p_ = torch.from_numpy(knn_p).float()
            for coeff in coeff_lst[1:]:
                new_p = EvalUtil.combine_knn_and_vocab_probs(
                            knn_p_,
                            p_,
                            coeff)
                ppl = EvalUtil.eval_ppl(new_p)
                #print('lim={} coeff={} ppl={}'.format(lim, coeff, ppl))
                if best_val is None or ppl < best_val:
                    best_val = ppl
                    best_knn_p = knn_p
                    best_cfg = (lim, coeff)
        out = {}
        out['tgts'] = tgts
        out['knn_tgts'] = knn_tgts
        out['dist'] = dist
        out['knn_p'] = best_knn_p
        out['p'] = p
        out['ppl'] = best_val
        out['index'] = index
        out['cfg'] = str(tuple(best_cfg))
        desc = self.cfg.copy()
        desc['n'] = p.shape[0]
        out['desc'] = 'knn-lm[{}]'.format(desc)
        if tag is not None:
            out['desc'] += '[{}]'.format(tag)
        return out




def main(args):
    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)
    dstore.add_exact(args.lookup, args.lookup_k)
    #dstore.add_annotations(args.dstore)

    tgts = dstore.tgts[:]
    knn_tgts = dstore.knn_tgts[:, :args.k]
    label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))
    print('')

    def print_results(out, baseline):
        if isinstance(out, (list, tuple)):
            for x in out:
                print_results(x)
            return
        diff = 0
        if baseline is not None:
            diff = out['ppl'] - baseline
        print('{:.4f} {:.4f} {:<16} {}'.format(diff, out['ppl'], out['cfg'], out['desc']))

    def find_occurs_gte_pivot(vocab, count):
        self = vocab
        for i, count_ in enumerate(self.count):
            if i < vocab.nspecial:
                continue
            if count >= count_:
                return i
        return None

    freq_list = [10**i for i in range(1, 8)]
    for freq in freq_list:
        piv = find_occurs_gte_pivot(vocab, freq)
        mask = np.logical_and(tgts >= vocab.nspecial, tgts < piv)
        lt = mask.sum().item()
        gte = (mask == False).sum().item()
        print('freq = {}, piv = {}, lt = {}, gte = {}'.format(freq, piv, lt, gte))
    print('')


    out_baseline = RunOriginal().run(dstore, vocab)
    baseline = out_baseline['ppl']
    print_results(out_baseline, None)
    res_approx = RunKNNLM(dict(k=1024, find_best_lim=False, use_exact=False, flip_distance=False, sort=True)).run(dstore, vocab)
    print_results(res_approx, baseline)
    res_exact  = RunKNNLM(dict(k=1024, find_best_lim=False, use_exact=True, flip_distance=True, sort=True)).run(dstore, vocab)
    print_results(res_exact, baseline)

    bin_0 = [None] + [10**i for i in range(1, 7)]
    bin_1 = bin_0[1:] + [None]
    sofar = 0
    print('max-count={}'.format(max(vocab.count)))
    print('len-vocab={}'.format(len(vocab)))
    coeff = 0.25
    for lo_freq, hi_freq in zip(bin_0, bin_1):
        if hi_freq is not None and lo_freq is not None:
            piv_start = find_occurs_gte_pivot(vocab, hi_freq + 1)
            piv_end = find_occurs_gte_pivot(vocab, lo_freq)
        elif hi_freq is not None:
            piv_start = find_occurs_gte_pivot(vocab, hi_freq + 1)
            piv_end = len(vocab)
        else:
            piv_start = vocab.nspecial
            piv_end = find_occurs_gte_pivot(vocab, lo_freq)
        assert piv_start < piv_end
        mask = np.logical_and(tgts >= piv_start, tgts <= piv_end)
        n = mask.sum().item()
        sofar += n
        # approx
        knn_p_, p_ = res_approx['knn_p'], res_approx['p']
        knn_p_, p_ = knn_p_[mask], p_[mask]
        knn_p_ = torch.from_numpy(knn_p_).float()
        p_ = torch.from_numpy(p_).float()
        approx_p = EvalUtil.combine_knn_and_vocab_probs(
                    knn_p_,
                    p_,
                    coeff)
        approx_ppl = EvalUtil.eval_ppl(approx_p).item()
        # exact
        knn_p_, p_ = res_exact['knn_p'], res_exact['p']
        knn_p_, p_ = knn_p_[mask], p_[mask]
        knn_p_ = torch.from_numpy(knn_p_).float()
        p_ = torch.from_numpy(p_).float()
        exact_p = EvalUtil.combine_knn_and_vocab_probs(
                    knn_p_,
                    p_,
                    coeff)
        exact_ppl = EvalUtil.eval_ppl(exact_p).item()
        # baseline
        baseline_ppl = EvalUtil.eval_ppl(p_).item()
        # main
        n = mask.sum().item()
        out = collections.OrderedDict()
        out['lo_freq'] = lo_freq
        out['hi_freq'] = hi_freq
        out['n'] = n
        out['approx-ppl'] = approx_ppl
        out['exact-ppl'] = exact_ppl
        out['baseline-ppl'] = baseline_ppl
        print(json.dumps(out))

        ######

        def pick(d, keys, mask=None):
            if mask is not None:
                return [d[k][mask] for k in keys]
            return [d[k] for k in keys]

        mask = mask.reshape(-1)
        tgts_ = tgts[mask]
        k = 128
        #
        index_a, dist_a, knn_tgts_a = pick(res_approx, ['index', 'dist', 'knn_tgts'], mask)
        index_e, dist_e, knn_tgts_e = pick(res_exact, ['index', 'dist', 'knn_tgts'], mask)

        res_overlap = collections.defaultdict(list)
        res_ppl = {}
        for k in [16, 64, 256]:
            for i in range(index_a.shape[0]):
                #a_, e_ = knn_tgts_a[i, :k].flatten().tolist(), knn_tgts_e[i, :k].flatten().tolist()
                a_, e_ = index_a[i, :k].flatten().tolist(), index_e[i, :k].flatten().tolist()
                overlap = len(set.intersection(set(a_), set(e_)))
                res_overlap[k].append(overlap)

        out = collections.OrderedDict()
        for k, v in res_overlap.items():
            out['overlap-{}'.format(k)] = np.mean(v)
        print(json.dumps(out))

        #print('piv=[{}:{}), freq=[{}:{}], n={}/{}, sofar={}'.format(
        #    piv_start, piv_end, lo_freq, hi_freq, n, mask.shape[0], sofar))

    sys.exit()


def edit_distance(x0, x1):
    m = len(x0)
    n = len(x1)
    d = [[i] for i in range(1, m + 1)]   # d matrix rows
    d.insert(0, list(range(0, n + 1)))   # d matrix columns
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if x0[i - 1] == x1[j - 1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i].insert(j, min(d[i - 1][j] + 1,
                               d[i][j - 1] + 1,
                               d[i - 1][j - 1] + substitutionCost))
    return d[-1][-1]


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self):
        path = self.path
        # self.keys = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
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
    parser.add_argument('--dstore', default='from_dstore_valid/va', type=str)
    parser.add_argument('--dstore-size', default=10000, type=int)
    parser.add_argument('--vocab', default='data-bin/wikitext-103/dict.txt')
    # dstore neighbors
    parser.add_argument('--lookup', default='from_dstore_valid/lookup_va', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    args = parser.parse_args()

    print(args)

    main(args)

