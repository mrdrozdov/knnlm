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

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from tqdm import tqdm

_my_globals = {}

def npy_copy(x):
    out = np.empty_like(x)
    out[:] = x
    return out


def main(args):
    use_cuda = torch.cuda.is_available()

    torch.set_grad_enabled(False)

    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)

    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize()

    k = 16 # TODO: Use full k.
    # TODO: Re-rank according to exact distance first.
    knns = npy_copy(dstore.knns[:, :k])
    dist = npy_copy(dstore.dist[:, :k])
    knn_tgts = npy_copy(dstore.knn_tgts[:, :k])
    prob = npy_copy(dstore.prob[:])
    vals = npy_copy(dstore.vals[:])
    tgts = npy_copy(dstore.tgts[:])

    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))

    train_tgts = npy_copy(knn_dstore.tgts[:])
    train_vals = train_tgts.copy()
    train_vals[1:] = train_tgts[:-1] # TODO: Missing 1.
    #train_tgts_ = torch.from_numpy(train_tgts)
    #knns_ = torch.from_numpy(knns)
    #if use_cuda:
    #    knns_ = knns_.cuda()
    #print(knns_.shape)
    #u, inv, counts = torch.unique(knns_, return_inverse=True, return_counts=True)
    u, inv = np.unique(knns, return_inverse=True)
    inv = inv.reshape(*knns.shape)

    # load model
    #hf_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    hf_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    hf_model = AutoModelForMaskedLM.from_pretrained('roberta-base')
    if use_cuda:
        hf_model.cuda()
    device = next(hf_model.parameters()).device

    output_dist = np.memmap('roberta_dist.npy', dtype=np.float32, mode='w+', shape=dist.shape)
    output_done = np.memmap('roberta_done.npy', dtype=np.int, mode='w+', shape=dist.shape)
    output_done[:] = 0

    # helper funcs
    def pick(d, keys):
        return [d[k] for k in keys]

    def get_batch_indices(batch_size, n, shuffle=False, verbose=False):
        assert shuffle == False
        num_batches = n // batch_size
        if num_batches * batch_size < n:
            num_batches += 1
        for i in tqdm(range(num_batches), disable=not verbose):
            start = i * batch_size
            end = min(start + batch_size, n)
            yield start, end

    def build_window(val, all_val, context_size=512, pad=0):
        pane = context_size // 2 - 1
        w_l = val.reshape(-1, 1) - 1 - np.arange(pane)[::-1].reshape(1, -1)
        w_r = val.reshape(-1, 1) + 1 + np.arange(pane).reshape(1, -1)
        w_mid = val.reshape(-1, 1)
        w = np.concatenate([w_l, w_mid, w_r], axis=1)
        tok = all_val[w.flatten()].reshape(*w.shape)
        tok[w < 0] = pad
        n = tok.shape[0]
        select_ids = np.array([pane] * n)
        return tok, select_ids

    def build_query_window(val, all_val, context_size=512, pad=0):
        val = val.flatten()
        pane = (context_size - val.shape[0]) // 2
        w_l = (val.min() - 1 - np.arange(pane)[::-1]).reshape(1, -1)
        w_r = (val.max() + 1 + np.arange(pane)).reshape(1, -1)
        w_mid = val.reshape(1, -1)
        w = np.concatenate([w_l, w_mid, w_r], axis=1)
        pad_mask = np.logical_or(w < 0, w >= all_val.shape[0])
        w[pad_mask] = 0
        tok = all_val[w.flatten()].reshape(*w.shape)
        tok[pad_mask] = pad
        select_ids = np.arange(w_mid.shape[1]) + w_l.shape[1]
        return tok, select_ids


    cache_tokens = {}
    def cached_tokenize(w):
        if w not in cache_tokens:
            cache_tokens[w] = hf_tokenizer.encode(w, add_special_tokens=False)
        return cache_tokens[w]

    def hf_tokenize(tokens):
        def to_text(tokens):
            return [vocab.symbols[tok] for tok in tokens]
        def to_hf_tokens(text):
            out = collections.defaultdict(list)
            for i_w, w in enumerate(text):
                #toks = hf_tokenizer.encode(w, add_special_tokens=False)
                toks = cached_tokenize(w)
                out['tokens'] += toks
                out['word_ids'] += [i_w] * len(toks)
            return out
        def pad_right(x, pad=-1):
            maxlen = max([len(xx) for xx in x])
            out = [xx + [pad] * (maxlen - len(xx)) for xx in x]
            return out

        batch_size, length = tokens.shape
        word_ids, hf_tokens = [], []
        for i_b in range(batch_size):
            text = to_text(tokens[i_b])
            word_ids_, hf_tokens_ = pick(to_hf_tokens(text), ['word_ids', 'tokens'])
            word_ids.append(word_ids_)
            hf_tokens.append(hf_tokens_)
        word_ids = torch.tensor(pad_right(word_ids, -1), dtype=torch.long)
        hf_tokens = torch.tensor(pad_right(hf_tokens, hf_tokenizer.pad_token_id), dtype=torch.long)
        return word_ids, hf_tokens

    def get_keys(hf_tokens, word_ids, select_ids):
        select_mask = word_ids == select_ids
        keys = []
        for start, end in get_batch_indices(batch_size=32, n=hf_tokens.shape[0]):
            hf_tokens_ = hf_tokens[start:end]
            if use_cuda:
                hf_tokens_ = hf_tokens_.cuda()
            model_output = hf_model(hf_tokens_, output_hidden_states=True)
            vecs = model_output['hidden_states'][-1]
            word_ids_ = word_ids[start:end]
            select_mask_ = select_mask[start:end]
            vecs[select_mask_[:, :, None].expand_as(vecs) == False] = 0
            keys_ = vecs.sum(1) / select_mask_.sum(1).view(-1, 1).to(vecs.device)
            keys.append(keys_)
        return torch.cat(keys, 0)

    def get_queries(hf_tokens, word_ids, select_ids, skip_agg=False):
        word_ids = word_ids.expand(select_ids.shape[0], word_ids.shape[1])
        if use_cuda:
            hf_tokens = hf_tokens.cuda()
        model_output = hf_model(hf_tokens, output_hidden_states=True)
        if skip_agg:
            return None
        vecs = model_output['hidden_states'][-1]
        vecs = vecs.expand(select_ids.shape[0], vecs.shape[1], vecs.shape[2]).clone()
        select_mask = word_ids == select_ids
        vecs[select_mask[:, :, None].expand_as(vecs) == False] = 0
        # TODO: Only sum where needed.
        queries = vecs.sum(1) / select_mask.sum(1).view(-1, 1).to(vecs.device)
        return queries

    if True:
        all_queries = []
        for start, end in get_batch_indices(batch_size=128, n=knns.shape[0], verbose=True):

            # Queries
            val_index = np.arange(start, end)
            tokens, select_ids = build_query_window(val_index, vals, context_size=256) # context size measured by knn-lm tokenization.
            select_ids = torch.from_numpy(select_ids).view(-1, 1)
            word_ids, hf_tokens = hf_tokenize(tokens)

            assert hf_tokens.shape[1] < hf_tokenizer.model_max_length

            queries = get_queries(hf_tokens, word_ids, select_ids)

            all_queries.append(queries.cpu())
        all_queries = torch.cat(all_queries, 0)

        u_offset = 0
        u_max = u[u_offset]
        for start, end in get_batch_indices(batch_size=128, n=train_vals.shape[0], verbose=True):

            batch_u, batch_u_offset = [], []
            if start <= u_max:
                while u_max < end:
                    batch_u.append(u_max)
                    batch_u_offset.append(u_offset)
                    u_offset += 1
                    u_max = u[u_offset]

            # Skip batch if not relevant.
            if len(batch_u) == 0:
                continue

            batch_u_index = np.array([idx - start for idx in batch_u])
            assert batch_u_index.min() >= 0

            # Keys
            val_index = np.arange(start, end)
            tokens, select_ids = build_query_window(val_index, train_vals, context_size=256) # context size measured by knn-lm tokenization.
            select_ids = select_ids[batch_u_index]
            select_ids = torch.from_numpy(select_ids).view(-1, 1)
            word_ids, hf_tokens = hf_tokenize(tokens)

            assert hf_tokens.shape[1] < hf_tokenizer.model_max_length

            keys = get_queries(hf_tokens, word_ids, select_ids)

            for i_u, u_ in enumerate(batch_u):
                offset_ = batch_u_offset[i_u]
                inv_mask = inv == offset_
                inv_mask_flat = inv_mask.any(axis=1).reshape(inv_mask.shape[0])
                key_ = keys[i_u].to(device)
                queries_ = all_queries[inv_mask_flat].to(device)
                d_ = -1 * torch.sum((queries_ - key_.unsqueeze(0))**2, dim=-1)
                output_dist[inv_mask] = d_.cpu().numpy()
                output_done[inv_mask] = 1
        import ipdb; ipdb.set_trace()
        pass


            # TODO: Find relevant keys.
            # TODO: Make a sparse update using distance.
            # new_dist_ = -1 * torch.sum((queries.unsqueeze(1) - keys.view(knns_.shape[0], knns_.shape[1], -1))**2, dim=-1)


    if False:
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min(start + batch_size, knns.shape[0])
            b = knns[start:end]
            b_ = torch.from_numpy(b)
            if use_cuda:
                b_ = b_.cuda()

            u, inv = torch.unique(b_, return_inverse=True)


    if False:
        for i in tqdm(range(num_slices), desc='encode'):
            start = i * slice_size
            end = min(start + slice_size, dataset_size)
            b_ = train_tgts_[start:end]
            if use_cuda:
                b_ = b_.cuda()

    import ipdb; ipdb.set_trace()
    pass


class EvalUtil:
    @staticmethod
    def get_knn_log_prob(tgts, dists, knn_tgts):

        tgts = torch.from_numpy(tgts).long().view(-1)
        dists = torch.from_numpy(dists).float().squeeze(-1)
        dists = -dists # lower distance should have more probability mass.
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
    args = parser.parse_args()

    print(args)

    main(args)

