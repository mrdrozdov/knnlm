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
import os

import numpy as np
import torch

from tqdm import tqdm


def main(args):
    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)

    tgts = dstore.tgts[:]
    # knns = dstore.knns[:, :args.k]
    knn_tgts = dstore.knn_tgts[:, :args.k]
    first_neighbor_is_tgt = tgts == knn_tgts[:, 0]
    print('{} / {} rows have first neighbor as target'.format(first_neighbor_is_tgt.sum(), tgts.shape[0]))
    print('Note: This could be less than 100% because approximate distance is used.')
    print('')

    # binary label indicating the knn target matches the original target
    label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    dsize = tgts.shape[0]
    has_positive = label.reshape(dsize, args.k).sum(axis=1) > 0
    has_negative = label.reshape(dsize, args.k).sum(axis=1) < args.k
    has_both = np.logical_and(has_positive, has_negative)
    print('has_positive = {} / {}'.format(np.sum(has_positive), dsize))
    print('has_negative = {} / {}'.format(np.sum(has_negative), dsize))
    print('has_both = {} / {}'.format(np.sum(has_both), dsize))
    print('')

    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))
    print('')

    nsample = 10
    sample_no_positive = tgts[has_positive == 0][:nsample]
    print('examples with no positives:')
    for i in range(nsample):
        idx = int(sample_no_positive[i])
        tok = vocab.symbols[idx]
        tf = vocab.count[idx]
        print('* [{}] "{}" with term frequency {}'.format(idx, tok, tf))
    print('')

    sample_no_negative = tgts[has_negative == 0][:nsample]
    print('examples with no negatives:')
    for i in range(nsample):
        idx = int(sample_no_negative[i])
        tok = vocab.symbols[idx]
        tf = vocab.count[idx]
        print('* [{}] "{}" with term frequency {}'.format(idx, tok, tf))
    print('')

    p = dstore.prob[:]
    dist = dstore.dist[:, :args.k]
    knn_p = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)

    p_ = torch.from_numpy(p).float()
    knn_p_ = torch.from_numpy(knn_p).float()
    coeff_lst = np.arange(20) / 20
    for coeff in coeff_lst:
        if coeff == 0:
            new_p = p_
        else:
            new_p = EvalUtil.combine_knn_and_vocab_probs(
                        knn_p_,
                        p_,
                        coeff)
        ppl = EvalUtil.eval_ppl(new_p)
        print('coeff = {:.3f}, knn_ppl = {}'.format(coeff, ppl))


    if args.optimal_2:

        print('With Optimal-2 Order')
        knn_p_optimal = EvalUtil.get_optimal_knn_log_prob(tgts, dist, knn_tgts)

        p_ = torch.from_numpy(p).float()
        knn_p_ = torch.from_numpy(knn_p_optimal).float()
        coeff_lst = np.arange(20) / 20
        for coeff in coeff_lst:
            if coeff == 0:
                new_p = p_
            else:
                new_p = EvalUtil.combine_knn_and_vocab_probs(
                            knn_p_,
                            p_,
                            coeff)
            ppl = EvalUtil.eval_ppl(new_p)
            print('coeff = {:.3f}, knn_ppl = {}'.format(coeff, ppl))




    if args.original:
        print('With Original Order')

        nsteps = 16
        stepsize = args.k // nsteps
        lim_lst = np.arange(nsteps)

        with open('coeff_lim_tradeoff-original.txt', 'w') as f:
            f.write('k lim coeff ppl\n')
            for i, lim in enumerate(lim_lst[1:]):
                lim = lim * stepsize
                print(i, lim)
                dist_ = dist[:, :lim]
                knn_tgts_ = knn_tgts[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)

                p_ = torch.from_numpy(p).float()
                knn_p_ = torch.from_numpy(knn_p).float()
                coeff_lst = np.arange(20) / 20
                for coeff in coeff_lst:
                    if coeff == 0:
                        new_p = p_
                    else:
                        new_p = EvalUtil.combine_knn_and_vocab_probs(
                                    knn_p_,
                                    p_,
                                    coeff)
                    ppl = EvalUtil.eval_ppl(new_p)

                    f.write('{} {} {} {}\n'.format(
                        args.k, lim, coeff, ppl))

    if args.optimal:
        print('With Optimal Order')
        optimal_order = Rerank.optimal_order(label, dist)

        nsteps = 16
        stepsize = args.k // nsteps
        lim_lst = np.arange(nsteps)

        with open('coeff_lim_tradeoff.txt', 'w') as f:
            f.write('k lim coeff ppl\n')
            for lim in lim_lst[1:]:
                lim = lim * stepsize
                print(lim)
                dist_ = np.take_along_axis(dist, optimal_order, axis=1)[:, :lim]
                knn_tgts_ = np.take_along_axis(knn_tgts, optimal_order, axis=1)[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)

                p_ = torch.from_numpy(p).float()
                knn_p_ = torch.from_numpy(knn_p).float()
                coeff_lst = np.arange(20) / 20
                for coeff in coeff_lst:
                    if coeff == 0:
                        new_p = p_
                    else:
                        new_p = EvalUtil.combine_knn_and_vocab_probs(
                                    knn_p_,
                                    p_,
                                    coeff)
                    ppl = EvalUtil.eval_ppl(new_p)

                    f.write('{} {} {} {}\n'.format(
                        args.k, lim, coeff, ppl))
    # # count all knns
    # def writefile(path, c):
    #     with open(path, 'w') as f:
    #         for idx in c.keys():
    #             tok = vocab.symbols[idx]
    #             f.write('{} {} {} {}\n'.format(
    #                 idx, tok, vocab.count[idx], c[idx]
    #                 ))

    # print('count neighbor frequency and write to file')
    # c = collections.Counter()
    # c.update(knn_tgts.reshape(-1))
    # writefile('knn_count.txt', c)

    print('')


class Rerank(object):
    @staticmethod
    def optimal_order(label, dist, big=1e6):
        n, k, _ = label.shape
        positive_dist = dist.copy()
        positive_dist[label == 0] = -np.inf
        positive_dist_sorted = np.sort(positive_dist, axis=1)[:, ::-1]
        positive_order = positive_dist.argsort(axis=1)[:, ::-1]

        ## negatives - sort from lo to hi
        negative_dist = dist.copy()
        negative_dist[label == 1] = -np.inf
        negative_dist_sorted = np.sort(negative_dist, axis=1)
        negative_order = negative_dist.argsort(axis=1)

        # set positives and negatives
        new_order = np.zeros((n, k, 1)).astype(np.int)
        new_order[positive_dist_sorted > -np.inf] = positive_order[positive_dist_sorted > -np.inf]
        new_order[negative_dist_sorted > -np.inf] = negative_order[negative_dist_sorted > -np.inf]

        assert np.all(np.unique(new_order, return_counts=True)[1] == n)

        return new_order



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
        # self.knns = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.knn_tgts = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))
        # self.lookup_done = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))


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
        def dist_func(d, k, q):
            return -1 * d

        tgts = torch.from_numpy(tgts).long().view(-1)
        dists = torch.from_numpy(dists).float().squeeze(-1)
        dists = -dists
        probs = torch.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(knn_tgts).long().squeeze(-1), tgts.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone().numpy()

        # Bx1
        return yhat_knn_prob.reshape(-1, 1)

    @staticmethod
    def get_optimal_knn_log_prob(tgts, dists, knn_tgts):
        k = knn_tgts.shape[1]

        lst = []
        for lim in range(1, k):
            lst.append(EvalUtil.get_knn_log_prob(tgts, dists[:, :lim], knn_tgts[:, :lim]))

        knn_p_ = np.concatenate(lst, axis=1)

        return knn_p_.argmin(axis=1).reshape(-1, 1)

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
    parser.add_argument('--dstore', default='from_dstore_valid/tr', type=str)
    parser.add_argument('--dstore-size', default=100000, type=int)
    parser.add_argument('--vocab', default='data-bin/wikitext-103/dict.txt')
    # dstore neighbors
    parser.add_argument('--lookup', default='from_dstore_valid/lookup_tr', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--optimal', action='store_true')
    parser.add_argument('--optimal-2', action='store_true')
    args = parser.parse_args()

    print(args)

    main(args)

