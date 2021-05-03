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

from vocab import Dictionary
from eval_util import *
from dstore_util import *
from dataset_util import *

import numpy as np
import torch
import sklearn.linear_model
import sklearn.preprocessing

from tqdm import tqdm

_my_globals = {}


class Experiment:



    def run(self, context):
        args = context['args']

        t = args.vocab_threshold

        trn_feat = context['train']['keys']
        trn_y = context['train']['tgts'] <= t
        trn_y = trn_y.astype(np.int).reshape(-1)

        tst_feat = context['test']['keys']
        tst_y = context['test']['tgts'] <= t
        tst_y = tst_y.astype(np.int).reshape(-1)

        solver = 'liblinear'
        model = sklearn.linear_model.LogisticRegression(max_iter=200, verbose=1, solver=solver)
        model.fit(trn_feat, trn_y)

        y_pred = model.predict(tst_feat)

        pos_acc = np.logical_and(tst_y == 1, y_pred == 1).sum() / np.sum(tst_y == 1)
        neg_acc = np.logical_and(tst_y == 0, y_pred == 0).sum() / np.sum(tst_y == 0)

        print('pos_n = {}, neg_n = {}'.format(np.sum(tst_y==1), np.sum(tst_y==0)))
        print('pos_acc = {:.3f}, neg_acc = {:.3f}'.format(pos_acc, neg_acc))

        mask_pos_pred = y_pred == 1
        mask_neg_pred = y_pred == 0
        mask_pos_gold = tst_y == 1
        mask_neg_gold = tst_y == 0

        out = {}
        out['mask_pos_pred'] = mask_pos_pred
        out['mask_neg_pred'] = mask_neg_pred
        out['mask_pos_gold'] = mask_pos_gold
        out['mask_neg_gold'] = mask_neg_gold
        return out




def main(args):
    context = DatasetUtils().build_context(args)

    #
    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=False)
    context['knn_dstore'] = knn_dstore

    #
    print('read vocab')
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens'.format(len(vocab)))
    print('')
    context['vocab'] = vocab

    # Add more.
    context['args'] = args

    #
    out = Experiment().run(context)

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
        print('n = {}, avg_k = {:.3f}'.format(tgts.shape[0], np.sum(knn_tgts >= 0, axis=1).mean()))
        print('ppl = {:.3f}, knn_ppl = {:.3f}'.format(ppl, new_ppl))

    def run_eval_mix(context, mask, skip_pos=False):
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
        mask_p = new_p.clone()
        mask_p[mask] = p_[mask]
        mask_ppl = EvalUtil.eval_ppl(mask_p)
        print('ppl = {:.3f}, knn_ppl = {:.3f}, mask_ppl = {:.3f}'.format(ppl, new_ppl, mask_ppl))

    # Unpack test data.
    knns = context['test']['knns']
    knn_tgts = context['test']['knn_tgts']
    tgts = context['test']['tgts']
    dist = context['test']['dist']
    p = context['test']['p']
    context_ = context

    keys = ['mask_pos_gold', 'mask_neg_gold']

    for k in keys:
        header = 'EVAL = {}'.format(k)
        print(header)
        print('-' * len(header))
        mask = out[k]
        context = {}
        context['knns'] = knns[mask]
        context['knn_tgts'] = knn_tgts[mask]
        context['tgts'] = tgts[mask]
        context['dist'] = dist[mask]
        context['p'] = p[mask]
        run_eval(context)

        print('')

    context = {}
    context['knns'] = knns
    context['knn_tgts'] = knn_tgts
    context['tgts'] = tgts
    context['dist'] = dist
    context['p'] = p

    # Overall with PRED and GOLD
    header = 'OVERALL'
    print(header)
    print('-' * len(header))

    print('EVAL PRED')
    m = out['mask_pos_pred']
    run_eval_mix(context, m, skip_pos=True)

    print('EVAL GOLD')
    m = out['mask_pos_gold']
    run_eval_mix(context, m, skip_pos=True)

    # Subset with PRED and GOLD
    header = 'SUBSET POS'
    print(header)
    print('-' * len(header))

    mask = out['mask_pos_gold']
    context = {}
    context['knns'] = knns[mask]
    context['knn_tgts'] = knn_tgts[mask]
    context['tgts'] = tgts[mask]
    context['dist'] = dist[mask]
    context['p'] = p[mask]

    print('EVAL PRED')
    m = out['mask_pos_pred'][mask]
    run_eval_mix(context, m, skip_pos=True)

    print('EVAL GOLD')
    m = out['mask_pos_gold'][mask]
    run_eval_mix(context, m, skip_pos=True)

    #
    header = 'SUBSET NEG'
    print(header)
    print('-' * len(header))

    mask = out['mask_neg_gold']
    context = {}
    context['knns'] = knns[mask]
    context['knn_tgts'] = knn_tgts[mask]
    context['tgts'] = tgts[mask]
    context['dist'] = dist[mask]
    context['p'] = p[mask]

    print('EVAL PRED')
    m = out['mask_pos_pred'][mask]
    run_eval_mix(context, m, skip_pos=True)

    print('EVAL GOLD')
    m = out['mask_pos_gold'][mask]
    run_eval_mix(context, m, skip_pos=True)


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
    parser.add_argument('--knn-dstore', default='dstore_train', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    # examine
    parser.add_argument('--k', default=1024, type=int)
    # output
    parser.add_argument('--vocab_threshold', default=1000, type=int)
    # debug
    parser.add_argument('--limit', default=-1, type=int)
    parser.add_argument('--preset', default='cross_valid', type=str)
    args = parser.parse_args()

    # Print flags.
    print(args)

    main(args)

