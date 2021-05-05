
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



def main(args):
    use_cuda = True
    device = torch.cuda.current_device() if use_cuda else None
    dkey = 'approx_dist' if args.approx else 'dist'

    context = DatasetUtils().build_context(args, include_keys=False)

    p = context['test']['p']
    dist = context['test'][dkey]
    tgts = context['test']['tgts']
    src = tgts.copy()
    src[1:] = tgts[:-1]
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

        print('n = {}, avg_k = {:.3f}, ppl = {:.3f}, knn_ppl = {:.3f}'.format(
            n, avg_k, ppl, new_ppl))

    def print_header(x, l=8, n=40):
        line = '-' * l
        line += ' ' + x + ' '
        line += '-' * (n - len(line))
        print(line)

    # COMPUTE ALL EVAL
    print_header('EVAL ALL')
    context = {}
    context['knns'] = knns
    context['knn_tgts'] = knn_tgts
    context['tgts'] = tgts
    context['dist'] = dist
    context['p'] = p
    run_eval(context)
    print('')

    # COMPUTE FILTERED EVAL

    w_counts = np.array(vocab.count).reshape(-1)
    src_counts = w_counts[src.reshape(-1)]
    tgt_counts = w_counts[tgts.reshape(-1)]

    t_list = [10**i for i in range(1, 7)]

    print('\n' + 't is src')

    for i in range(0, len(t_list)):
        t0 = t_list[i]
        t1 = np.inf if (i+1) >= len(t_list) else t_list[i+1]

        mask = np.logical_and(src_counts >= t0, src_counts < t1)

        print_header('interval = {}:{}'.format(t0, t1))

        context = {}
        context['knns'] = knns[mask]
        context['knn_tgts'] = knn_tgts[mask]
        context['tgts'] = tgts[mask]
        context['dist'] = dist[mask]
        context['p'] = p[mask]
        run_eval(context)

        print('')

    print('\n' + 't is tgt')

    for i in range(0, len(t_list)):
        t0 = t_list[i]
        t1 = np.inf if (i+1) >= len(t_list) else t_list[i+1]

        mask = np.logical_and(tgt_counts >= t0, tgt_counts < t1)

        print_header('interval = {}:{}'.format(t0, t1))

        context = {}
        context['knns'] = knns[mask]
        context['knn_tgts'] = knn_tgts[mask]
        context['tgts'] = tgts[mask]
        context['dist'] = dist[mask]
        context['p'] = p[mask]
        run_eval(context)

        print('')

    print('\n' + 'use both')

    for i in range(0, len(t_list)):
        for j in range(0, len(t_list)):
            t0_src = t_list[i]
            t1_src = np.inf if (i+1) >= len(t_list) else t_list[i+1]

            t0_tgt = t_list[j]
            t1_tgt = np.inf if (j+1) >= len(t_list) else t_list[j+1]

            mask_src = np.logical_and(src_counts >= t0_src, src_counts < t1_src)
            mask_tgt = np.logical_and(tgt_counts >= t0_tgt, tgt_counts < t1_tgt)
            mask = np.logical_and(mask_src, mask_tgt)

            print_header('interval = src:{}:{} tgt:{}:{}'.format(t0_src, t1_src, t0_tgt, t1_tgt))

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

