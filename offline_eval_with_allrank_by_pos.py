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
import sys

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
        self.flip_distance = False
        self.flip_score = False
        self.use_coeff = False
        for k, v in cfg.items():
            setattr(self, k, v)

    def run(self, dstore, vocab, mask=None, mask_b=None, tag=None):
        if mask is None:
            p = dstore.prob[:].copy()
            dist = dstore.dist[:, :self.k].copy()
            knn_tgts = dstore.knn_tgts[:, :self.k].copy()
            tgts = dstore.tgts[:].copy()
        else:
            p = dstore.prob[:].copy()[mask]
            dist = dstore.dist[:, :self.k].copy()[mask]
            knn_tgts = dstore.knn_tgts[:, :self.k].copy()[mask]
            tgts = dstore.tgts[:].copy()[mask]

        if self.flip_distance:
            dist = -dist

        p_ = torch.from_numpy(p).float()
        original_ppl = EvalUtil.eval_ppl(p_)

        best_val = None
        best_cfg = None
        limits_to_check = 8
        limit_size = self.k // limits_to_check
        limits_to_check_lst = [(i + 1) * limit_size for i in range(limits_to_check)]
        if not self.find_best_lim:
            limits_to_check_lst = [self.k]
        coeff_lst = np.arange(20) / 20
        if self.use_coeff:
            coeff = dstore.allrank.coeff[:].copy()[mask_b]

        if self.use_coeff:
            for lim in limits_to_check_lst:
                dist_ = dist[:, :lim]
                knn_tgts_ = knn_tgts[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)
                knn_p_ = torch.from_numpy(knn_p).float()

                new_p = EvalUtil.combine_knn_and_vocab_probs(
                            knn_p_,
                            p_,
                            np.clip(coeff.round(), 0.01, 0.2))
                ppl = EvalUtil.eval_ppl(new_p)

                if best_val is None or ppl < best_val:
                    best_val = ppl
                    best_cfg = (lim, 'use')

        else:
            for lim in limits_to_check_lst:
                dist_ = dist[:, :lim]
                knn_tgts_ = knn_tgts[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)
                knn_p_ = torch.from_numpy(knn_p).float()
                for coeff in coeff_lst[1:]:
                    new_p = EvalUtil.combine_knn_and_vocab_probs(
                                knn_p_,
                                p_,
                                coeff.item())
                    ppl = EvalUtil.eval_ppl(new_p)
                    #print('lim={} coeff={} ppl={}'.format(lim, coeff, ppl))
                    if best_val is None or ppl < best_val:
                        best_val = ppl
                        best_cfg = (lim, coeff)
        if False:
            for lim in limits_to_check_lst:
                dist_ = dist[:, :lim]
                knn_tgts_ = knn_tgts[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)
                knn_p_ = torch.from_numpy(knn_p).float()
                for coeff in coeff_lst[1:]:
                    new_p = EvalUtil.combine_knn_and_vocab_probs(
                                knn_p_,
                                p_,
                                coeff.item())
                    ppl = EvalUtil.eval_ppl(new_p)
                    #print('lim={} coeff={} ppl={}'.format(lim, coeff, ppl))
                    if best_val is None or ppl < best_val:
                        best_val = ppl
                        best_cfg = (lim, coeff)
        out = {}
        out['ppl'] = best_val
        out['cfg'] = str(tuple(best_cfg))
        desc = self.cfg.copy()
        desc['n'] = p.shape[0]
        out['desc'] = 'knn-lm[{}]'.format(desc)
        if tag is not None:
            out['desc'] += '[{}]'.format(tag)
        return out


class RunRerank:
    def __init__(self, cfg):
        self.cfg = cfg
        # defaults
        self.rerank = False
        self.use_scores = False
        self.flip_distance = False
        self.flip_score = False
        self.use_coeff = False
        # override
        for k, v in cfg.items():
            setattr(self, k, v)

    def run(self, dstore, vocab, mask=None, mask_b=None, tag=None):
        if mask is None:
            p = dstore.prob[:].copy()
            dist = dstore.dist[:, :self.k].copy()
            knn_tgts = dstore.knn_tgts[:, :self.k].copy()
            tgts = dstore.tgts[:].copy()
        else:
            p = dstore.prob[:].copy()[mask]
            dist = dstore.dist[:, :self.k].copy()[mask]
            knn_tgts = dstore.knn_tgts[:, :self.k].copy()[mask]
            tgts = dstore.tgts[:].copy()[mask]
        if self.flip_distance:
            dist = -dist
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)
        if self.rerank:
            new_order = dstore.allrank.knn_rank[:][mask_b]
            assert new_order.shape[0] == dist.shape[0]
            dist = np.take_along_axis(dist, new_order, axis=1)
            knn_tgts = np.take_along_axis(knn_tgts, new_order, axis=1)
        if self.use_scores:
            dist = dstore.allrank.scores[:, :self.k].copy()[mask_b]
            if self.flip_score:
                dist = -dist
        if self.use_coeff:
            coeff = dstore.allrank.coeff[:].copy()[mask_b]

        p_ = torch.from_numpy(p).float()
        original_ppl = EvalUtil.eval_ppl(p_)

        best_val = None
        best_cfg = None
        limits_to_check = 8
        limit_size = self.k // limits_to_check
        limits_to_check_lst = [(i + 1) * limit_size for i in range(limits_to_check)]
        if not self.find_best_lim:
            limits_to_check_lst = [self.k]
        coeff_lst = np.arange(20) / 20

        if self.use_coeff:
            for lim in limits_to_check_lst:
                dist_ = dist[:, :lim]
                knn_tgts_ = knn_tgts[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)
                knn_p_ = torch.from_numpy(knn_p).float()

                new_p = EvalUtil.combine_knn_and_vocab_probs(
                            knn_p_,
                            p_,
                            np.clip(coeff.round(), 0.01, 0.2))
                ppl = EvalUtil.eval_ppl(new_p)

                if best_val is None or ppl < best_val:
                    best_val = ppl
                    best_cfg = (lim, 'use')

        else:
            for lim in limits_to_check_lst:
                dist_ = dist[:, :lim]
                knn_tgts_ = knn_tgts[:, :lim]
                knn_p = EvalUtil.get_knn_log_prob(tgts, dist_, knn_tgts_)
                knn_p_ = torch.from_numpy(knn_p).float()
                for coeff in coeff_lst[1:]:
                    new_p = EvalUtil.combine_knn_and_vocab_probs(
                                knn_p_,
                                p_,
                                coeff.item())
                    ppl = EvalUtil.eval_ppl(new_p)
                    #print('lim={} coeff={} ppl={}'.format(lim, coeff, ppl))
                    if best_val is None or ppl < best_val:
                        best_val = ppl
                        best_cfg = (lim, coeff)
        out = {}
        out['ppl'] = best_val
        out['cfg'] = str(tuple(best_cfg))
        desc = self.cfg.copy()
        desc['n'] = p.shape[0]
        out['desc'] = 'rerank[{}]'.format(desc)
        if tag is not None:
            out['desc'] += '[{}]'.format(tag)
        return out

class RunOptimal:
    def __init__(self, cfg):
        self.cfg = cfg
        # defaults
        self.rerank = False
        self.use_scores = False
        self.flip_distance = False
        self.flip_score = False
        # override
        for k, v in cfg.items():
            setattr(self, k, v)

    def run(self, dstore, vocab, mask=None, mask_b=None, tag=None):
        if mask is None:
            p = dstore.prob[:].copy()
            dist = dstore.dist[:, :self.k].copy()
            knn_tgts = dstore.knn_tgts[:, :self.k].copy()
            tgts = dstore.tgts[:].copy()
        else:
            p = dstore.prob[:].copy()[mask]
            dist = dstore.dist[:, :self.k].copy()[mask]
            knn_tgts = dstore.knn_tgts[:, :self.k].copy()[mask]
            tgts = dstore.tgts[:].copy()[mask]
        if self.flip_distance:
            dist = -dist
        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)
        new_order = Rerank.optimal_order(label, dist)
        dist = np.take_along_axis(dist, new_order, axis=1)
        knn_tgts = np.take_along_axis(knn_tgts, new_order, axis=1)
        if self.rerank:
            new_order = dstore.allrank.knn_rank[:][mask_b]
            assert new_order.shape[0] == dist.shape[0]
            dist = np.take_along_axis(dist, new_order, axis=1)
            knn_tgts = np.take_along_axis(knn_tgts, new_order, axis=1)
        if self.use_scores:
            dist = dstore.allrank.scores[:, :self.k].copy()[mask_b]
            if self.flip_score:
                dist = -dist

        p_ = torch.from_numpy(p).float()
        original_ppl = EvalUtil.eval_ppl(p_)

        best_val = None
        best_cfg = None
        limits_to_check = 8
        limit_size = self.k // limits_to_check
        limits_to_check_lst = [(i + 1) * limit_size for i in range(limits_to_check)]
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
                            coeff.item())
                ppl = EvalUtil.eval_ppl(new_p)
                #print('lim={} coeff={} ppl={}'.format(lim, coeff, ppl))
                if best_val is None or ppl < best_val:
                    best_val = ppl
                    best_cfg = (lim, coeff)
        out = {}
        out['ppl'] = best_val
        out['cfg'] = str(tuple(best_cfg))
        desc = self.cfg.copy()
        desc['n'] = p.shape[0]
        out['desc'] = 'optimal[{}]'.format(desc)
        if tag is not None:
            out['desc'] += '[{}]'.format(tag)
        return out


def main(args):
    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)
    dstore.add_allrank(args.allrank, args.allrank_size, args.allrank_k)
    if args.pos:
        dstore.add_annotations(args.dstore)

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

    query_id = dstore.allrank.query_id[:].reshape(-1)
    print('WARNING: Manually setting query id.')
    query_id = query_id - 100000
    assert query_id.min() >= 0

    freq_lst = [10, 100, 1000, 10000, -1][::-1]
    target_tag_lst = [None]

    if args.pos:
        src_pos = dstore.src_pos[:]
        tgt_pos = np.concatenate([src_pos[1:], np.zeros(1).reshape(1, 1)], axis=0) # HACK
        target_tag_lst.append(['NN', 'NNP', 'NNPS', 'NNS'])
        target_tag_lst.append(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    for target_tag in target_tag_lst:
        target_tag_idx = [dstore.tag2idx[t] for t in target_tag] if target_tag is not None else None

        for top_freq in freq_lst:
            print('RESULTS, top_freq={}, tag={}'.format(top_freq, target_tag))

            skip_tags = False
            mask = np.zeros(tgts.shape[0], dtype=np.int)
            mask[query_id] = 1
            mask = mask == 1
            if top_freq > 0:
                is_top_freq = np.logical_and(tgts.reshape(-1) >= 4, tgts.reshape(-1) < top_freq + 4)
                mask = np.logical_and(mask, is_top_freq)
            if target_tag_idx is not None and not skip_tags:
                is_tag = np.isin(tgt_pos.reshape(-1), target_tag_idx)
                mask_ = np.logical_and(mask, is_tag)
                if mask_.sum() == 0:
                    skip_tags = True
                    print('WARNING: Skipping tags.')
                else:
                    mask = mask_
            assert mask.sum() > 0


            mask_b = np.ones(query_id.shape[0], dtype=int) == 1
            tgts_ = tgts[query_id]
            if top_freq > 0:
                mask_b[tgts_.reshape(-1) < 4] = False
                mask_b[tgts_.reshape(-1) >= top_freq + 4] = False
            if target_tag_idx is not None and not skip_tags:
                tgt_pos_ = tgt_pos[query_id]
                is_tag = np.isin(tgt_pos_.reshape(-1), target_tag_idx)
                mask_ = np.logical_and(mask_b, is_tag)
                if mask_.sum() == 0:
                    skip_tags = True
                    print('WARNING: Skipping tags.')
                else:
                    mask_b = mask_
            assert mask_b.sum() > 0

            #TODO: Add more checks for mask and mask_b.

            fd = args.flip_distance
            fs = args.flip_score

            out_baseline = RunOriginal().run(dstore, vocab, mask=mask, mask_b=mask_b)
            baseline = out_baseline['ppl']
            print_results(out_baseline, None)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=1024, find_best_lim=False)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=1024, find_best_lim=False, use_coeff=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=1024, find_best_lim=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=1024, find_best_lim=True, use_coeff=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=False)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunKNNLM(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=True, use_coeff=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            #print_results(RunOptimal(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunRerank(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=True, rerank=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunRerank(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=True, rerank=True, use_scores=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)
            print_results(RunRerank(dict(flip_distance=fd, flip_score=fs, k=args.allrank_k, find_best_lim=True, rerank=True, use_scores=True, use_coeff=True)).run(dstore, vocab, mask=mask, mask_b=mask_b), baseline)

    sys.exit()



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


    def add_allrank(self, path, size, k):
        self.allrank = DstoreAllrank.load(path, size, k)


class DstoreAllrank:
    @staticmethod
    def load(path, size, k):
        out = DstoreAllrank()
        out.knn_tgts = np.memmap(os.path.join(path, 'out_vali_knn_tgts.npy'), dtype=np.int, mode='r', shape=(size, k, 1))
        out.knns = np.memmap(os.path.join(path, 'out_vali_knns.npy'), dtype=np.int, mode='r', shape=(size, k, 1))
        out.knn_rank = np.memmap(os.path.join(path, 'out_vali_knn_rank.npy'), dtype=np.int, mode='r', shape=(size, k, 1))
        out.scores = np.memmap(os.path.join(path, 'out_vali_scores.npy'), dtype=np.float32, mode='r', shape=(size, k, 1))
        out.query_id = np.memmap(os.path.join(path, 'out_vali_query_id.npy'), dtype=np.int, mode='r', shape=(size, 1))
        out.coeff = np.memmap(os.path.join(path, 'out_vali_coeff.npy'), dtype=np.float32, mode='r', shape=(size, 1))
        return out


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
        if isinstance(coeff, (int, float)):
            coeff = torch.FloatTensor(1).fill_(coeff)
        elif not isinstance(coeff, torch.Tensor):
            coeff = torch.from_numpy(coeff).float()
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = torch.log(1 - coeff)
        coeffs[1] = torch.log(coeff)
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
    # allrank
    parser.add_argument('--allrank', default='allrank_output/results/exp_acl-full-ranknet-2-epoch1', type=str)
    parser.add_argument('--allrank-size', default=7856, type=int)
    parser.add_argument('--allrank-k', default=128, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    parser.add_argument('--k-freq', default=-1, type=int)
    parser.add_argument('--flip-distance', action='store_true')
    parser.add_argument('--flip-score', action='store_true')
    parser.add_argument('--pos', action='store_true')
    args = parser.parse_args()

    print(args)

    main(args)

