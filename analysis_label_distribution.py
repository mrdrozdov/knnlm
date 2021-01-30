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
import pandas as pd

from tqdm import tqdm


def main(args):
    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)

    dsize = dstore.tgts.shape[0]
    done_mask = (dstore.lookup_done[:] == 1).reshape(-1) # boolean mask must be flat
    ndone = done_mask.sum()

    print('according to lookup, {} / {} rows are populated'.format(ndone, dsize))
    print('')

    print('loading targets. this may take a few minutes')
    tgts = dstore.tgts[done_mask][:]
    print('loading knns. this may take a few minutes')
    knns = dstore.knns[done_mask][:, :args.k][:]
    knn_tgts = dstore.knn_tgts[done_mask][:, :args.k][:]
    knns0_tgt = knn_tgts[:, 0]
    first_neighbor_is_tgt = tgts == knns0_tgt
    print('{} / {} rows have first neighbor as target'.format(first_neighbor_is_tgt.sum(), tgts.shape[0]))
    print('')

    # binary label indicating the knn target matches the original target
    label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    has_positive = label.reshape(ndone, args.k).sum(axis=1) > 0
    has_negative = label.reshape(ndone, args.k).sum(axis=1) < args.k
    has_both = np.logical_and(has_positive, has_negative)
    print('has_positive = {} / {}'.format(np.sum(has_positive), ndone))
    print('has_negative = {} / {}'.format(np.sum(has_negative), ndone))
    print('has_both = {} / {}'.format(np.sum(has_both), ndone))
    print('')

    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))
    print('')

    nsample = 10
    tgts_ = tgts[has_positive == 0]
    index = np.arange(tgts_.shape[0])
    np.random.shuffle(index)
    index = index[:nsample]
    sample_no_positive = tgts_[index]
    print('examples with no positives:')
    for i in range(nsample):
        idx = int(sample_no_positive[i])
        tok = vocab.symbols[idx]
        tf = vocab.count[idx]
        print('* [{}] "{}" with term frequency {}'.format(idx, tok, tf))
    print('')

    tgts_ = tgts[has_negative == 0]
    index = np.arange(tgts_.shape[0])
    np.random.shuffle(index)
    index = index[:nsample]
    sample_no_negative = tgts_[index]
    print('examples with no negatives:')
    for i in range(nsample):
        idx = int(sample_no_negative[i])
        tok = vocab.symbols[idx]
        tf = vocab.count[idx]
        print('* [{}] "{}" with term frequency {}'.format(idx, tok, tf))
    print('')

    # count all knns
    if False:
        def writefile(path, c):
            with open(path, 'w') as f:
                for idx in c.keys():
                    tok = vocab.symbols[idx]
                    f.write('{} {} {} {}\n'.format(
                        idx, tok, vocab.count[idx], c[idx]
                        ))

        print('count neighbor frequency and write to file')
        c = collections.Counter()
        c.update(knn_tgts.reshape(-1))
        writefile('knn_count.txt', c)

        print('count tp and write to file')
        c = collections.Counter()
        c.update(knn_tgts[label == 1].reshape(-1))
        writefile('tp_count.txt', c)

        print('count fp and write to file')
        c = collections.Counter()
        c.update(knn_tgts[label == 0].reshape(-1))
        writefile('fp_count.txt', c)

        print('count npos and write to file')
        c = collections.Counter()
        for i in range(ndone):
            idx = int(tgts[i])
            npos = label[i].sum()
            c[idx] += npos
        writefile('npos_count.txt', c)
        print('')

    # Count unique entries.
    if False:
        def print_unique(knns, knn_tgts, cfg=None):
            num_keys = np.unique(knns).shape[0]
            num_vocab = np.unique(knn_tgts).shape[0]
            if cfg is None:
                print('vocab={} keys={}'.format(num_vocab, num_keys))
            else:
                print('[{}] vocab={} keys={}'.format(cfg, num_vocab, num_keys))

        print_unique(knns, knn_tgts)

        num_k_splits = 4
        num_top_lst = [-1, 0, 10, 100]
        num_top_start = np.argmax(vocab.count)
        for top in num_top_lst:
            top = num_top_start + top if top >= 0 else top
            for i_k in range(num_k_splits):
                k_ = (i_k + 1) * (args.k // num_k_splits)

                knns_ = knns[:, :k_]
                knn_tgts_ = knn_tgts[:, :k_]
                top_mask = knn_tgts_ >= top
                knn_tgts_ = knn_tgts_[top_mask]
                knns_ = knns_[top_mask]

                print_unique(knns_, knn_tgts_, cfg=(top, k_))
        print('')

    # term freq X key freq X len(unique(keys))
    # unique_terms = np.unique(knn_tgts)
    # for term in tqdm(unique_terms):
    #     mask = knn_tgts == term
    #     knns_ = knns[mask]
    term_to_key = collections.defaultdict(set)
    term_count = collections.Counter()
    for term, key in tqdm(zip(knn_tgts.reshape(-1).tolist(), knns.reshape(-1))):
        term_to_key[term].add(key)
        term_count[term] += 1
def w_(term_to_key, term_count, vocab):
    with open('tf_by_kf_by_uniq.txt', 'w') as f:
        terms = list(term_count.keys())
        f.write('sym tf tf_as_key kf\n')
        for t in terms:
            sym = vocab.symbols[t]
            tf = vocab.count[t]
            kf = len(term_to_key[t])
            tf_as_key = term_count[t]
            f.write('{} {} {} {}\n'.format(
                sym, tf, tf_as_key, kf))
    pass
    # 3058601.44it/s
    # 1101144.43it/s
    # df = pd.DataFrame({'term': knn_tgts.reshape(-1), 'key': knns.reshape(-1), 'ones': np.ones(knns.reshape(-1).shape[0])})
    import ipdb; ipdb.set_trace()
    pass




class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self):
        path = self.path
        self.keys = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.tgts = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.prob = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        self._initialized = True

    def add_neighbors(self, path, k):
        self.knns = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.knn_tgts = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))
        self.lookup_done = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))


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
    parser.add_argument('--k', default=64, type=int)
    args = parser.parse_args()

    print(args)

    main(args)

