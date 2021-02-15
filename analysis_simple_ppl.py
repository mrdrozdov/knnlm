import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

from tqdm import tqdm


class RunOriginal:
    def run(self, dstore):
        p = dstore.prob[:].copy()
        p_ = torch.from_numpy(p).float()
        ppl = EvalUtil.eval_ppl(p_).item()
        out = {}
        out['ppl'] = ppl
        out['cfg'] = None
        return out


class RunKNNLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.flip_distance = False
        for k, v in cfg.items():
            setattr(self, k, v)

    def run(self, dstore):
        p = dstore.prob[:].copy()
        dist = dstore.dist[:, :self.k].copy()
        if self.flip_distance:
            dist = -dist
        knn_tgts = dstore.knn_tgts[:, :self.k].copy()
        tgts = dstore.tgts[:].copy()

        p_ = torch.from_numpy(p).float()
        original_ppl = EvalUtil.eval_ppl(p_)

        best_val = None
        best_cfg = None
        coeff_lst = np.arange(20) / 20

        knn_p = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)
        knn_p_ = torch.from_numpy(knn_p).float()
        for coeff in coeff_lst[1:]:
            new_p = EvalUtil.combine_knn_and_vocab_probs(
                        knn_p_,
                        p_,
                        coeff)
            ppl = EvalUtil.eval_ppl(new_p)
            if best_val is None or ppl < best_val:
                best_val = ppl.item()
                best_cfg = coeff.item()
        out = {}
        out['ppl'] = best_val
        out['cfg'] = best_cfg
        out['desc'] = self.cfg.copy()
        return out


def main(args):
    dstore = Dstore(args.dstore, args.dstore_size, 1024)
    dstore.initialize()
    dstore.add_neighbors(args.lookup, args.lookup_k)

    out = RunOriginal().run(dstore)
    print(json.dumps(out))
    print('original ppl = {}'.format(out['ppl']))
    out = RunKNNLM(dict(k=1024, flip_distance=args.flip_distance)).run(dstore)
    print(json.dumps(out))
    print('knnlm ppl = {}'.format(out['ppl']))


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None):
        self.path = path
        self.dstore_size = dstore_size
        self.vec_size = vec_size
        self._initialized = False

    def initialize(self):
        path = self.path
        self.tgts = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.prob = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, 1))
        self._initialized = True

    def add_neighbors(self, path, k):
        self.knn_tgts = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='r', shape=(self.dstore_size, k, 1))
        self.dist = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, k, 1))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='from_dstore_valid/va', type=str)
    parser.add_argument('--dstore-size', default=10000, type=int)
    # dstore neighbors
    parser.add_argument('--lookup', default='from_dstore_valid/lookup_va', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    parser.add_argument('--flip-distance', action='store_true')
    args = parser.parse_args()

    print(args)

    main(args)

