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

import spacy
from spacy.tokens import Doc
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

    special_tokens = set()
    special_tokens.update([vocab.bos_index, vocab.eos_index, vocab.pad_index, vocab.unk_index])
    special_tokens.update([vocab.index('=')])
    special_tokens_counter = collections.Counter()

    src = dstore.src[:]
    docs = []
    doc = []
    for i, idx in enumerate(src.reshape(-1).tolist()):
        doc.append(idx)
        if idx == vocab.eos_index:
            docs.append(doc)
            doc = []
        if idx in special_tokens:
            special_tokens_counter[idx] += 1
    if len(doc) > 0:
        docs.append(doc)
    print('# sentences = {}'.format(len(docs)))
    print('max sent length = {}'.format(max([len(x) for x in docs])))

    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    tag_counter = collections.Counter()
    src_tags = []
    for idx_lst in tqdm(docs):
        tokens = [vocab.symbols[idx] for idx in idx_lst]
        text = ' '.join(tokens)
        doc = nlp(text)
        assert len(doc) == len(tokens), (tokens, doc.text)
        tags = [x.tag_ for x in doc]
        src_tags += tags
        tag_counter.update(tags)
    assert len(src_tags) == src.shape[0], (len(src_tags), src.shape)

    for idx in sorted(special_tokens):
        sym = vocab.symbols[idx]
        val = special_tokens_counter[idx]
        print('{:>12} {:>12} {:>6}'.format(idx, sym, val))

    path = os.path.join(args.dstore, 'pos_dict.txt')
    print('writing part-of-speech dictionary {}'.format(path))
    write_pos_dict(path, tag_counter)

    path = os.path.join(args.dstore, 'annotation_src_pos.npy')
    print('writing part-of-speech annotations {}'.format(path))
    tag2idx = {k: i for i, k in enumerate(sorted(tag_counter.keys()))}
    src_tag_idx_lst = [tag2idx[idx] for idx in src_tags]
    out = np.memmap(path, mode='w+', dtype=np.int, shape=src.shape)
    out[:] = np.array(src_tag_idx_lst).reshape(*src.shape)


def write_pos_dict(path, tag_counter):
    with open(path, 'w') as f:
        for i, k in enumerate(sorted(tag_counter.keys())):
            f.write('{} {} {}\n'.format(i, k, tag_counter[k]))



class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


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
        self.src = np.memmap(os.path.join(path, 'dstore_src.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
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

