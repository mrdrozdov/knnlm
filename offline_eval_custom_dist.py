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

import numpy as np
import torch

from tqdm import tqdm

_my_globals = {}

def pick(d, keys):
    return [d[k] for k in keys]

def npy_copy(x):
    out = np.empty_like(x)
    out[:] = x
    return out


class RunOriginal:
    def run(self, p):
        p_ = torch.from_numpy(p).float()
        ppl = EvalUtil.eval_ppl(p_)
        out = {}
        out['ppl'] = ppl
        return out


class RunKNNLM:
    def run(self, p, tgts, dist, knn_tgts, coeff=0.25):

        p_ = torch.from_numpy(p).float()
        original_ppl = EvalUtil.eval_ppl(p_)

        knn_p = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)
        knn_p_ = torch.from_numpy(knn_p).float()
        new_p = EvalUtil.combine_knn_and_vocab_probs(knn_p_, p_, coeff)
        new_ppl = EvalUtil.eval_ppl(new_p)

        out = {}
        out['knn_p'] = knn_p
        out['new_p'] = new_p
        out['original_ppl'] = original_ppl
        out['new_ppl'] = new_ppl
        return out




def main(args):
    use_cuda = torch.cuda.is_available()

    torch.set_grad_enabled(False)

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)
    dstore.add_dist(args.custom_dist, args.custom_k, args.custom_size)

    cdist = npy_copy(dstore.custom_dist)
    cdone = npy_copy(dstore.custom_done)
    limit, k = cdist.shape[:2]
    is_done = cdone.sum()
    is_done_numel = cdone.shape[0] * cdone.shape[1]
    done_row_mask = cdone.any(axis=1).flatten()
    print('limit = {} ({}), k = {}, done = {}/{} ({:.3f})'.format(
        limit, done_row_mask.sum(), k, is_done, is_done_numel, is_done/is_done_numel))

    cdist = cdist[done_row_mask]
    cdone = cdone[done_row_mask]

    tgts = npy_copy(dstore.tgts[:limit][done_row_mask])
    knn_tgts = npy_copy(dstore.knn_tgts[:limit, :k][done_row_mask])
    knns = npy_copy(dstore.knns[:limit, :k][done_row_mask])
    dist = npy_copy(dstore.dist[:limit, :k][done_row_mask])
    prob = npy_copy(dstore.prob[:limit][done_row_mask])

    # original
    out = RunOriginal().run(prob)
    print('# ORIGINAL')
    print('original {:.3f}'.format(out['ppl']))
    print('')

    def sort_and_truncate(dist, new_k, extra):
        index = np.argsort(dist, axis=1)[:, ::-1]
        new_extra = {}
        for k in list(extra.keys()):
            v = extra[k]
            new_v = np.take_along_axis(v, index, axis=1)
            new_extra[k] = new_v[:, :new_k]

        return new_extra

    new_k = 8
    for coeff in [0.1]:
        extra = {}
        extra['dist'] = dist
        extra['knn_tgts'] = knn_tgts

        print('@' * 20)
        print('coeff = {}'.format(coeff))
        # knn-lm
        out = RunKNNLM().run(prob, tgts, dist, knn_tgts, coeff=coeff)
        print('# KNN-LM')
        print('original {:.3f}'.format(out['original_ppl']))
        print('   knnlm {:.3f}'.format(out['new_ppl']))
        print('')

        # knn-lm (done)
        dist_ = dist.copy()
        dist_[cdone == 0] = -10000
        out = RunKNNLM().run(prob, tgts, dist_, knn_tgts, coeff=coeff)
        print('# KNN-LM (done)')
        print('original {:.3f}'.format(out['original_ppl']))
        print('   knnlm {:.3f}'.format(out['new_ppl']))
        print('')

        # knn-lm (done)
        dist_ = cdist.copy()
        dist_[cdone == 0] = -10000
        out = RunKNNLM().run(prob, tgts, dist_, knn_tgts, coeff=coeff)
        print('# KNN-LM (custom)')
        print('original {:.3f}'.format(out['original_ppl']))
        print('   knnlm {:.3f}'.format(out['new_ppl']))
        print('')

        # knn-lm (done)
        dist_ = cdist.copy()
        dist_ = -1 * dist_
        dist_[cdone == 0] = -10000
        out = RunKNNLM().run(prob, tgts, dist_, knn_tgts, coeff=coeff)
        print('# KNN-LM (custom * -1)')
        print('original {:.3f}'.format(out['original_ppl']))
        print('   knnlm {:.3f}'.format(out['new_ppl']))
        print('')

        dist_ = cdist.copy()
        dist_[cdone == 0] = -10000
        dist_ = dist_[:, :new_k]
        knn_tgts_ = knn_tgts[:, :new_k]
        out = RunKNNLM().run(prob, tgts, dist_, knn_tgts_, coeff=coeff)
        print('# KNN-LM (trunc)')
        print('original {:.3f}'.format(out['original_ppl']))
        print('   knnlm {:.3f}'.format(out['new_ppl']))
        print('')

        dist_ = cdist.copy()
        dist_[cdone == 0] = -10000
        new_extra = sort_and_truncate(dist_, new_k, extra)
        dist_, knn_tgts_ = pick(new_extra, ['dist', 'knn_tgts'])
        out = RunKNNLM().run(prob, tgts, dist_, knn_tgts_, coeff=coeff)
        print('# KNN-LM (trunc and sort)')
        print('original {:.3f}'.format(out['original_ppl']))
        print('   knnlm {:.3f}'.format(out['new_ppl']))
        print('')

    sys.exit()

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

    def add_dist(self, path, k, size=None):
        if size is None:
            size = self.dstore_size
        self.custom_dist = np.memmap('{}_dist.npy'.format(path), dtype=np.float32, mode='r', shape=(size, k, 1))
        self.custom_done = np.memmap('{}_done.npy'.format(path), dtype=np.int, mode='r', shape=(size, k, 1))


class Dictionary(object):
    """
    A mapping from symbols to consecutive integers.

    Taken from fairseq repo.
    """

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = collections.Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        for line in f.readlines():
            idx = line.rfind(" ")
            if idx == -1:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt>'"
                )
            word = line[:idx]
            count = int(line[idx + 1 :])
            self.indices[word] = len(self.symbols)
            self.symbols.append(word)
            self.count.append(count)


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
    parser.add_argument('--dstore', default='dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    parser.add_argument('--vocab', default='data-bin/wikitext-103/dict.txt')
    # dstore neighbors
    parser.add_argument('--lookup', default='dstore_valid/lookup', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    parser.add_argument('--custom-dist', default='./roberta', type=str)
    parser.add_argument('--custom-k', default=16, type=int)
    parser.add_argument('--custom-size', default=20000, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    args = parser.parse_args()

    print(args)

    main(args)

