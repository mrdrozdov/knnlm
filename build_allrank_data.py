import argparse
import collections
import os

import faiss

import numpy as np

from tqdm import tqdm


def main(args):
    np.random.seed(args.seed)

    os.system('mkdir -p {}'.format(args.output))

    print('TRAIN')
    out = build_split(args.tr_dstore, args.tr_dstore_size, args.tr_lookup, args.tr_lookup_k, args.k, args.ntrain,
            shuffle=not args.tr_noshuffle, balance=args.tr_balance, should_filter_top=True, require_both=True)
    path = os.path.join(args.output, 'train.txt')
    write_allrank_data(path, out['qids'], out['label'], out['feat_idx'], out['q_src'], out['q_tgt'], out['feat_tgt'])

    print('VALID')
    out = build_split(args.va_dstore, args.va_dstore_size, args.va_lookup, args.va_lookup_k, args.k, args.nvalid,
            shuffle=args.va_shuffle, balance=False)
    path = os.path.join(args.output, 'vali.txt')
    write_allrank_data(path, out['qids'], out['label'], out['feat_idx'], out['q_src'], out['q_tgt'], out['feat_tgt'])


def build_split(split_dstore_path, split_dstore_size, lookup_path, lookup_k, k, n, shuffle=True, balance=False, should_filter_top=False, require_both=False):
    split_dstore = Dstore(split_dstore_path, split_dstore_size, 1024)
    split_dstore.initialize(has_row=True)
    split_dstore.add_neighbors(lookup_path, lookup_k)

    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()

    index = np.arange(split_dstore.tgts.shape[0])
    if shuffle:
        np.random.shuffle(index)
    index = index[:n]

    src = split_dstore.src[index]
    tgts = split_dstore.tgts[index]
    knns = split_dstore.knns[index, :lookup_k]
    dist = split_dstore.dist[index, :lookup_k] # dist is sorted from hi to lo
    knn_tgts = split_dstore.knn_tgts[index, :lookup_k]
    qids = split_dstore.row[index]
    size = qids.shape[0]
    label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    if should_filter_top and args.filter_top > 0:
        print('FILTER TOP {}'.format(args.filter_top))
        # assumes vocab is sorted desc after special tokens
        tf = np.array(vocab.count)
        tgts_tf = tf[tgts]
        top_N_abs_tf_lim = tf[np.argmax(tf) + args.filter_top]
        mask = (tgts_tf < top_N_abs_tf_lim).reshape(-1)
        print('will filter {} / {}'.format(mask.sum(), mask.shape[0]))

        src = src[mask]
        tgts = tgts[mask]
        knns = knns[mask]
        dist = dist[mask]
        knn_tgts = knn_tgts[mask]
        qids = qids[mask]
        size = qids.shape[0]
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    if balance:
        npos = 10
        nneg = k - npos

        # count stats
        print('STATS (BEFORE BALANCE)')
        has_positive = label.reshape(size, lookup_k).sum(axis=1) >= npos
        has_negative = label.reshape(size, lookup_k).sum(axis=1) < lookup_k - nneg
        has_both = np.logical_and(has_positive, has_negative)
        print('has enough positive = {} / {}'.format(np.sum(has_positive), size))
        print('has enough negative = {} / {}'.format(np.sum(has_negative), size))
        print('has_both = {} / {}'.format(np.sum(has_both), size))

        order = np.arange(lookup_k).reshape(1, -1).repeat(size, axis=0).reshape(size, lookup_k, 1).astype(np.int)

        pos_order = order.copy()
        pos_order[label == 0] = lookup_k + 10**6
        pos_order.sort(axis=1)
        pos_order = pos_order[:, :npos]
        assert pos_order[has_both].max() < lookup_k

        neg_order = order.copy()
        neg_order[label == 1] = lookup_k + 10**6
        neg_order.sort(axis=1)
        neg_order = neg_order[:, :nneg]
        assert neg_order[has_both].max() < lookup_k

        def _filter(x, i0, i1):
            x0 = np.take_along_axis(x[has_both], i0[has_both], axis=1)
            x1 = np.take_along_axis(x[has_both], i1[has_both], axis=1)
            return np.concatenate([x0, x1], axis=1)

        tgts = tgts[has_both]
        src = src[has_both]
        knns = _filter(knns, pos_order, neg_order)
        dist = _filter(dist, pos_order, neg_order)
        knn_tgts = _filter(knn_tgts, pos_order, neg_order)
        qids = qids[has_both]
        size = qids.shape[0]
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)


    else:
        knns = knns[:, :k]
        dist = dist[:, :k]
        knn_tgts = knn_tgts[:, :k]
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

    # count stats
    print('STATS')
    has_positive = label.reshape(size, k).sum(axis=1) > 0
    has_negative = label.reshape(size, k).sum(axis=1) < k
    has_both = np.logical_and(has_positive, has_negative)
    print('has_positive = {} / {}'.format(np.sum(has_positive), size))
    print('has_negative = {} / {}'.format(np.sum(has_negative), size))
    print('has_both = {} / {}'.format(np.sum(has_both), size))

    if require_both:
        qids = qids[has_both]
        dist = dist[has_both]
        label = label[has_both]
        knns = knns[has_both]
        tgts = tgts[has_both]
        src = src[has_both]
        knn_tgts = knn_tgts[has_both]

    size = label.shape[0]

    # get optimal order
    print('get optimal order')

    ## positives - sort from hi to lo
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
    new_order = np.zeros((size, k, 1)).astype(np.int)
    new_order[positive_dist_sorted > -np.inf] = positive_order[positive_dist_sorted > -np.inf]
    new_order[negative_dist_sorted > -np.inf] = negative_order[negative_dist_sorted > -np.inf]

    assert np.all(np.unique(new_order, return_counts=True)[1] == size)

    print('re-order')
    label = np.take_along_axis(label, new_order, axis=1)
    knns = np.take_along_axis(knns, new_order, axis=1)
    knn_tgts = np.take_along_axis(knn_tgts, new_order, axis=1)

    out = {}
    out['qids'] = qids
    out['label'] = label
    out['feat_idx'] = knns
    out['feat_tgt'] = knn_tgts
    out['q_src'] = src
    out['q_tgt'] = tgts


    return out


def write_allrank_data(path, qids, label, feat_idx, query_src, query_tgt, feat_tgt):
    print('writing {} with feat_idx shape {}'.format(path, feat_idx.shape))
    size, k, _ = label.shape
    with open(path, 'w') as f:
        for i_slate in range(size):
            for i_k in range(k):
                q_id = int(qids[i_slate])
                y = int(label[i_slate, i_k, 0])
                x_id = feat_idx[i_slate, i_k, 0]
                q_src = int(query_src[i_slate])
                q_tgt = int(query_tgt[i_slate])
                x_tgt = int(feat_tgt[i_slate, i_k, 0])
                f.write('{} qid:{} 0:{} 1:{} 2:{} 3:{}'.format(y, q_id, x_id, q_src, q_tgt, x_tgt))
                f.write('\n')


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


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self, has_row=False):
        path = self.path
        self.keys = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.tgts = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.src = np.memmap(os.path.join(path, 'dstore_src.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.prob = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        if has_row:
            self.row = np.memmap(os.path.join(path, 'dstore_row.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self._initialized = True

    def add_neighbors(self, path, k):
        self.knns = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.knn_tgts = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))
        self.lookup_done = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dstore
    #parser.add_argument('--dstore', default='dstore_train', type=str)
    #parser.add_argument('--dstore-size', default=103225485, type=int)
    # dstore-tr
    parser.add_argument('--tr-dstore', default='from_dstore_valid-2/tr', type=str)
    parser.add_argument('--tr-dstore-size', default=100000, type=int)
    parser.add_argument('--tr-lookup', default='from_dstore_valid-2/lookup_tr', type=str)
    parser.add_argument('--tr-lookup-k', default=1024, type=int)
    parser.add_argument('--ntrain', default=100000, type=int)
    parser.add_argument('--tr-noshuffle', action='store_true')
    parser.add_argument('--tr-balance', action='store_true')
    # dstore-va
    parser.add_argument('--va-dstore', default='from_dstore_valid-2/va', type=str)
    parser.add_argument('--va-dstore-size', default=100000, type=int)
    parser.add_argument('--va-lookup', default='from_dstore_valid-2/lookup_va', type=str)
    parser.add_argument('--va-lookup-k', default=1024, type=int)
    parser.add_argument('--nvalid', default=100000, type=int)
    parser.add_argument('--va-shuffle', action='store_true')
    # allrank
    parser.add_argument('--filter-top', default=-1, type=int, help='If > 1, then removes any example with target in the top N term freqs.')
    parser.add_argument('--vocab', default='data-bin/wikitext-103/dict.txt', type=str)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--seed', default=1231, type=int)
    parser.add_argument('--k', default=256, type=int)
    args = parser.parse_args()

    print(args)

    main(args)

