
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(x, c='HEADER'):
    print(getattr(bcolors, c) + x + bcolors.ENDC)

def print_color_mid(*args, **kwargs):
    print(to_color_mid(*args, **kwargs))

def to_color(x, c='HEADER'):
    return getattr(bcolors, c) + x + bcolors.ENDC

def to_color_mid(l, x, r, c='HEADER'):
    return l + getattr(bcolors, c) + x + bcolors.ENDC + r


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

    print('load eval')
    context = DatasetUtils().build_context(args, include_keys=False)

    print('load train')
    knn_dstore = Dstore(args.knn_dstore, args.knn_dstore_size, 1024)
    knn_dstore.initialize(include_keys=False)
    trn_tgts = npy_copy(knn_dstore.vals[:])

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

        print('n = {}, avg_k = {:.3f}, ppl = {:.3f}, knn_ppl = {:.3f}'.format(
            n, avg_k, ppl, new_ppl))

    def print_header(x, l=8, n=40):
        line = '-' * l
        line += ' ' + x + ' '
        line += '-' * (n - len(line))
        print(line)

    print_header('eval ppl')
    context = {}
    context['knns'] = knns
    context['knn_tgts'] = knn_tgts
    context['tgts'] = tgts
    context['dist'] = dist
    context['p'] = p
    run_eval(context)
    print('\n\n')

    # COMPUTE ALL EVAL
    logp = p
    p = np.exp(logp)
    knn_logp = EvalUtil.get_knn_log_prob(tgts, dist, knn_tgts)
    knn_p = np.exp(knn_logp)
    argmax = np.argmax(dist, axis=1).flatten()

    if args.sort == 'diff':
        index = np.argsort(knn_p.reshape(-1) - p.reshape(-1))[::-1]
    elif args.sort == 'argmax':
        index = np.argsort(np.argmax(dist, axis=1).flatten())[::-1]
    elif args.sort == 'inv_diff':
        index = np.argsort(p.reshape(-1) - knn_p.reshape(-1))[::-1]

    def tostring(x):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if not isinstance(x, (list, tuple)):
            x = [x]
        s = ' '.join([vocab.symbols[tok] for tok in x])
        return s

    def pretty_eval_context(i, lsize=32, rsize=8):
        l = tgts[i-lsize:i].reshape(-1)
        x = tgts[i].reshape(-1)
        r = tgts[i+1:i+1+rsize].reshape(-1)
        return to_color_mid(tostring(l) + ' ', tostring(x), ' ' + tostring(r), c='BOLD')

    def pretty_key_context(i, lsize=32, rsize=8):
        l = trn_tgts[i-lsize:i].reshape(-1)
        x = trn_tgts[i].reshape(-1)
        r = trn_tgts[i+1:i+1+rsize].reshape(-1)
        return to_color_mid(tostring(l) + ' ', tostring(x), ' ' + tostring(r), c='BOLD')

    def word_overlap(i, past=1024):
        tgt_context = set(tgts[i-past:i].flatten().tolist())
        wo = np.zeros(1024, dtype=np.int)

        for i_j, j in enumerate(knns[i].flatten().tolist()):
            trn_context = set(trn_tgts[j-past:j].flatten().tolist())
            wo[i_j] = len(set.intersection(tgt_context, trn_context))
        return wo

    for i in index.tolist():
        tgt = tgts[i].item()
        kp_ = max(knn_p[i].item(), 1e-4)
        p_ = max(p[i].item(), 1e-4)
        diff = kp_ - p_
        k_ppl = 2**(-np.log(kp_) / np.log(2))
        p_ppl = 2**(-np.log(p_) / np.log(2))

        correct = np.sum(knn_tgts[i] == tgt)
        total = knn_tgts.shape[1]
        acc = correct  / total

        below_threshold = tgt <= 1000
        w = tostring(tgt)
        wo = word_overlap(i)

        # prob mass
        dist_ = torch.from_numpy(dist[i].reshape(-1))
        prob  = torch.softmax(dist_, 0)
        logp =  torch.log_softmax(dist_, 0)
        entropy = - torch.sum(prob * logp).item()
        argmax_ = argmax[i].item()
        valmax_ = prob[argmax_].item()

        if valmax_ < 0.1:
            continue

        line ='{:>12} diff={:.3f} kp={:.3f}[{:.3f}] p={:.3f}[{:.3f}] acc={:.3f}[{}/{}] ent={:.3f} argmax={} valmax={:.3f} below_threshold={} w={}'.format(
            tgt, diff, kp_, k_ppl, p_, p_ppl,
            acc, correct, total, entropy, argmax_, valmax_, below_threshold, w,
            )

        print(line)
        print('-' * len(line))
        # eval context
        line = pretty_eval_context(i)
        print(line)

        # key context
        k = 1024
        for j in range(k):
            knn = knns[i, j].item()
            tgt_ = trn_tgts[knn].item()
            pmass = prob[j].item()
            matches = tgt == tgt_
            if not matches and j > 16:
                if pmass < 0.001:
                    continue
            wo_ = wo[j].item()
            mbit = to_color('1', 'OKGREEN') if matches else to_color('0', 'FAIL')
            line = pretty_key_context(knn)
            line = '{:>4}. {:>012} : {} : {:>4} : {:.3f} : '.format(j, knn, mbit, wo_, pmass) + line
            print(line)

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
    #
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
    parser.add_argument('--preset', default='valid', type=str)
    parser.add_argument('--sort', default='diff', type=str)
    parser.add_argument('--approx', action='store_true')
    args = parser.parse_args()

    # Print flags.
    print(args)

    main(args)

