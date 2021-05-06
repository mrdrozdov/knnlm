
import argparse
import collections
import json
import os
import sys

from vocab import Dictionary
from eval_util import *
from dstore_util import *
from dataset_util import *
from ngram_util import *

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
    src = make_src(tgts, args.max_ngram)
    knn_tgts = context['test']['knn_tgts']
    knns = context['test']['knns']

    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=False)

    trn_tgt = npy_copy(knn_dstore.tgts[:])
    trn_src = make_src(trn_tgt, args.max_ngram)
    ngram_lm = build_ngram_lm(trn_src, trn_tgt, limit=args.ngram_limit, max_ngram=args.max_ngram)

    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    vocab_size = len(vocab)
    print('found {} tokens'.format(len(vocab)))
    print('')

    def run_eval(context):
        coeff = context['coeff']
        knns = context['knns']
        knn_tgts = context['knn_tgts']
        src = context['src']
        tgts = context['tgts']
        dist = context['dist']
        p = context['p']

        def rerank():
            ngram_dist = np.zeros(knns.shape, dtype=np.float32)

            k = knn_tgts.shape[1]
            for i_k in range(k):
                tgts_ = knn_tgts[:, i_k]
                ngram_dist[:, i_k] = ngram_lm.batch_predict(src, tgts_)

            # Use a stable sort since there are many ties.
            ngram_index = np.argsort(ngram_dist, axis=1, kind='mergesort')[:, ::-1]

            def take_(x):
                return np.take_along_axis(x, ngram_index, axis=1)

            new_k = 512

            res = {}
            res['knns'] = take_(knns)[:, :new_k]
            res['knn_tgts'] = take_(knn_tgts)[:, :new_k]
            res['dist'] = take_(dist)[:, :new_k]
            return res

        res = rerank()
        knns = res['knns']
        knn_tgts = res['knn_tgts']
        dist = res['dist']

        label = (knn_tgts == tgts.reshape(-1, 1, 1)).astype(np.int)

        knn_p = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)

        ngram_p = ngram_lm.batch_predict(src, tgts)
        ngram_logp = np.log(np.clip(ngram_p, 1e-8, 1))
        ngram_logp[ngram_p == 0] = -10000
        del ngram_p

        p_ = torch.from_numpy(p).float()
        knn_p_ = torch.from_numpy(knn_p).float()
        ngram_p_ = torch.from_numpy(ngram_logp).float()

        unif_p_ = p_.clone()
        unif_p_[:] = np.log(1/vocab_size)

        #
        unif_knn_p = EvalUtil.combine_knn_and_vocab_probs(
                    knn_p_,
                    unif_p_,
                    0.9)
        unif_ngram_p = EvalUtil.combine_knn_and_vocab_probs(
                    ngram_p_,
                    unif_p_,
                    0.9)

        interp_knn_p = EvalUtil.combine_knn_and_vocab_probs(
                    knn_p_,
                    p_,
                    coeff)
        interp_ngram_p = EvalUtil.combine_knn_and_vocab_probs(
                    ngram_p_,
                    p_,
                    coeff)

        interp_many_p = EvalUtil.combine_many_probs(
                [knn_p_, ngram_p_],
                p_,
                [0.1, 0.2]
                )

        ppl = EvalUtil.eval_ppl(p)
        knn_ppl = EvalUtil.eval_ppl(unif_knn_p)
        ngram_ppl = EvalUtil.eval_ppl(unif_ngram_p)
        interp_knn_ppl = EvalUtil.eval_ppl(interp_knn_p)
        interp_ngram_ppl = EvalUtil.eval_ppl(interp_ngram_p)
        interp_many_ppl = EvalUtil.eval_ppl(interp_many_p)

        n = tgts.shape[0]

        print('n = {}, coeff = {:.3f}, ppl = {:.3f}, knn_ppl = {:.3f}, ngram_ppl = {:.3f}, i_knn_ppl = {:.3f}, i_ngram_ppl = {:.3f}, i_many_ppl = {:.3f}'.format(
            n, coeff, ppl, knn_ppl, ngram_ppl, interp_knn_ppl, interp_ngram_ppl, interp_many_ppl))

    def print_header(x, l=8, n=40):
        line = '-' * l
        line += ' ' + x + ' '
        line += '-' * (n - len(line))
        print(line)

    # COMPUTE ALL EVAL
    print_header('EVAL ALL')
    context = {}
    context['coeff'] = 0.25
    context['knns'] = knns
    context['knn_tgts'] = knn_tgts
    context['src'] = src
    context['tgts'] = tgts
    context['dist'] = dist
    context['p'] = p
    run_eval(context)
    print('')

    print_header('many coeff')
    coeff_list = np.arange(1, 10) / 20
    for coeff in coeff_list:
        context['coeff'] = coeff
        context['knns'] = knns
        context['knn_tgts'] = knn_tgts
        context['src'] = src
        context['tgts'] = tgts
        context['dist'] = dist
        context['p'] = p
        run_eval(context)
    print('')

    # COMPUTE FILTERED EVAL

    w_counts = np.array(vocab.count).reshape(-1)
    src_counts = w_counts[src[:, -1].reshape(-1)]
    tgt_counts = w_counts[tgts.reshape(-1)]

    t_list = [10**i for i in range(1, 7)]

    print('\n' + 't is src')

    for i in range(0, len(t_list)):
        t0 = t_list[i]
        t1 = np.inf if (i+1) >= len(t_list) else t_list[i+1]

        mask = np.logical_and(src_counts >= t0, src_counts < t1)

        print_header('interval = {}:{}'.format(t0, t1))

        context = {}
        context['coeff'] = 0.25
        context['knns'] = knns[mask]
        context['knn_tgts'] = knn_tgts[mask]
        context['src'] = src[mask]
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
        context['coeff'] = 0.25
        context['knns'] = knns[mask]
        context['knn_tgts'] = knn_tgts[mask]
        context['src'] = src[mask]
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
            context['coeff'] = 0.25
            context['knns'] = knns[mask]
            context['knn_tgts'] = knn_tgts[mask]
            context['src'] = src[mask]
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
    # dstore
    parser.add_argument('--knn-dstore', default='dstore_train', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    # dstore neighbors
    parser.add_argument('--lookup', default='dstore_valid/lookup', type=str)
    parser.add_argument('--lookup-k', default=1024, type=int)
    # dstore
    parser.add_argument('--vocab_threshold', default=1000, type=int)

    # examine
    parser.add_argument('--k', default=1024, type=int)
    # debug
    parser.add_argument('--limit', default=-1, type=int)
    parser.add_argument('--ngram-limit', default=-1, type=int, help='Number of training examples for ngram lm.')
    parser.add_argument('--max-ngram', default=2, type=int, help='Token width of ngram lm.')
    parser.add_argument('--preset', default='valid', type=str)
    parser.add_argument('--approx', action='store_true')
    args = parser.parse_args()

    # Print flags.
    print(args)

    main(args)

