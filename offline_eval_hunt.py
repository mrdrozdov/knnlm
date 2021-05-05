
import argparse
import collections
import json
import os
import sys

from vocab import Dictionary
from eval_util import *
from dstore_util import *
from dataset_util import *

import numpy as np
import torch

from tqdm import tqdm


class FilterUtils:
    @staticmethod
    def get_keep_mask(knns, knn_tgts, keep_ids, batch_size=1024):
        mask = np.full_like(knns, False, dtype=np.bool)
        n = knns.shape[0]
        for i in tqdm(range(0, n, batch_size), desc='get-mask-keep'):
            b = knns[i:min(i + batch_size, n)]
            b_tgts = knn_tgts[i:min(i + batch_size, n)]
            m = np.isin(b, keep_ids)
            # Don't discard targets above the threshold.
            #m = np.logical_or(m, b_tgts >= args.vocab_threshold)
            mask[i:min(i + batch_size, n)] = m
        return mask


def main(args):
    use_cuda = True
    device = torch.cuda.current_device() if use_cuda else None
    dkey = 'approx_dist' if args.approx else 'dist'

    context = DatasetUtils().build_context(args, include_keys=False)

    p = context['test']['p']
    dist = context['test'][dkey]
    tgts = context['test']['tgts']
    knn_tgts = context['test']['knn_tgts']
    knns = context['test']['knns']

    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))
    print('')

    def run_eval(context):
        knns = context['knns']
        knn_tgts = context['knn_tgts']
        tgts = context['tgts']
        dist = context['dist']
        p = context['p']

        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

        knn_p = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)

        coeff = 0.25
        p_ = torch.from_numpy(p).float()
        knn_p_ = torch.from_numpy(knn_p).float()
        new_p = EvalUtil.combine_knn_and_vocab_probs(
                    knn_p_,
                    p_,
                    coeff)
        ppl = EvalUtil.eval_ppl(p)
        new_ppl = EvalUtil.eval_ppl(new_p)
        n = tgts.shape[0]
        avg_k = np.sum(knn_tgts >= 0, axis=1).mean()

        import ipdb; ipdb.set_trace()

        print('n = {}, avg_k = {:.3f}, ppl = {:.3f}, knn_ppl = {:.3f}'.format(
            n, avg_k, ppl, new_ppl))

    def print_header(header):
        print(header)
        print('-' * len(header))

    # COMPUTE ALL EVAL
    print_header('EVAL ALL')
    mask = np.full_like(tgts, True, dtype=np.bool).reshape(-1)
    context = {}
    context['knns'] = knns[mask]
    context['knn_tgts'] = knn_tgts[mask]
    context['tgts'] = tgts[mask]
    context['dist'] = dist[mask]
    context['p'] = p[mask]
    run_eval(context)
    print('')

    # COMPUTE FILTERED EVAL

    print_header('EVAL <= T')
    mask = (tgts <= args.vocab_threshold).reshape(-1)
    context = {}
    context['knns'] = knns[mask]
    context['knn_tgts'] = knn_tgts[mask]
    context['tgts'] = tgts[mask]
    context['dist'] = dist[mask]
    context['p'] = p[mask]
    run_eval(context)
    print('')

    print_header('EVAL > T')
    mask = (tgts > args.vocab_threshold).reshape(-1)
    context = {}
    context['knns'] = knns[mask]
    context['knn_tgts'] = knn_tgts[mask]
    context['tgts'] = tgts[mask]
    context['dist'] = dist[mask]
    context['p'] = p[mask]
    run_eval(context)
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dstore
    parser.add_argument('--dstore', default='dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    parser.add_argument('--vocab', default='data-bin/wikitext-103/dict.txt')
    #
    parser.add_argument('--test-dstore', default='dstore_test', type=str)
    parser.add_argument('--test-dstore-size', default=245569, type=int)
    parser.add_argument('--test-lookup', default='dstore_test/lookup')
    # dstore neighbors
    parser.add_argument('--lookup', default='dstore_valid/lookup', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # dstore
    parser.add_argument('--vocab_threshold', default=1000, type=int)

    # examine
    parser.add_argument('--k', default=1024, type=int)
    # debug
    parser.add_argument('--limit', default=-1, type=int)
    parser.add_argument('--preset', default='valid', type=str)
    parser.add_argument('--approx', action='store_true')
    args = parser.parse_args()

    # Print flags.
    print(args)

    main(args)

