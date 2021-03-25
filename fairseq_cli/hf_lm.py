#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Build datastore using a huggingface model.
"""

import collections
import copy
import logging
import math
import os
import json

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.knnlm import KNN_Dstore


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm')


def invert(lst):
    d = collections.defaultdict(list)
    for x in lst:
        for k, v in x.items():
            d[k].append(v)
    return d


class Writer:
    def __init__(self, outdir, max_size=-1, k=-1, vec_size=1024):
        # TODO: Write metadata. Should record last offset.
        self.outdir = os.path.abspath(outdir)
        self.initialized = False
        self.done = False
        self.fp = {}
        self.dtypes= {
                'target': [max_size],
                'queries': [max_size, vec_size],
                'dists': [max_size, k],
                'knns': [max_size, k],
                'keys': [max_size, k, vec_size],
                'vals': [max_size, k],
                'probs': [max_size],
        }
        self.max_size = max_size
        self.k = k
        self.vec_size = vec_size
        self.offset = 0

    def initialize(self):
        print('Initializing...')
        # Make directory.
        outdir = self.outdir
        try:
            os.system('mkdir -p {}'.format(outdir))
        except:
            pass

        # Open arrays.
        for k, shape in self.dtypes.items():
            shape = tuple(shape)
            outfile = os.path.join(outdir, '{}.npy'.format(k))
            self.fp[k] = np.memmap(outfile, dtype=np.float, mode='w+', shape=shape)

        self.initialized = True

    def update(self, o):
        if not self.initialized:
            self.initialize()
        if self.done:
            return
        offset = self.offset
        size = None
        for k in self.dtypes.keys():
            v = o[k]
            m = self.fp[k]
            if size is None:
                size = v.shape[0]
                if self.offset + size > self.max_size:
                    size = self.max_size - self.offset
            m[offset:offset+size] = v[:size]
        self.offset += size
        if self.offset >= self.max_size:
            self.done = True
            print('Done! Filled reference data with {} items.'.format(self.offset))

    def close(self):
        # TODO
        raise NotImplementedError


def collate(save_extra):
    def helper(extra_lst):
        new_extra = collections.defaultdict(list)
        for extra in extra_lst:
            for k, v in extra.items():
                new_extra[k].append(v)
        for k, v in new_extra.items():
            new_extra[k] = np.concatenate(v, 0)
        return new_extra
    return helper(save_extra)


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args):
    if parsed_args.dstore_mmap is not None:
        d = os.path.dirname(parsed_args.dstore_mmap)
        print('mmap from {}'.format(d))
        if not os.path.exists(d):
            print('making dir')
            os.system('mkdir -p {}'.format(d))

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load model.
    hf_tokenizer = AutoTokenizer.from_pretrained(parsed_args.hf_model)
    if parsed_args.hf_enc_mode == 'masked':
        hf_model = AutoModelForMaskedLM.from_pretrained(parsed_args.hf_model)
    elif parsed_args.hf_enc_mode == 'causal':
        hf_model = AutoModelForCausalLM.from_pretrained(parsed_args.hf_model)

    args = copy.deepcopy(parsed_args)

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    task_dataset = task.dataset(args.gen_subset)
    assert args.context_window > 0
    dataset = LMContextWindowDataset(
        dataset=task_dataset,
        tokens_per_sample=args.tokens_per_sample,
        context_window=args.context_window,
        pad_idx=task.source_dictionary.pad(),
    )
    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            parsed_args.hf_max_position
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    #).next_epoch_itr(shuffle=True)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    if args.knnlm and args.save_knnlm_dstore:
        raise ValueError("Cannot use knnlm while trying to build the datastore!")

    if args.knnlm:
        knn_dstore = KNN_Dstore(args)

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        if args.save_knnlm_dstore:
            print('keytype being saved:', args.knn_keytype)
            if args.dstore_fp16:
                print('Saving fp16')
                dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_prob = np.memmap(args.dstore_mmap+'_prob.npy', dtype=np.float16, mode='w+', shape=(args.dstore_size, 1))
                dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='w+', shape=(args.dstore_size, 1))
                dstore_tgts = np.memmap(args.dstore_mmap+'_tgts.npy', dtype=np.int16, mode='w+', shape=(args.dstore_size, 1))
                dstore_src = np.memmap(args.dstore_mmap+'_src.npy', dtype=np.int16, mode='w+', shape=(args.dstore_size, 1))
            else:
                print('Saving fp32')
                dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='w+', shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_prob = np.memmap(args.dstore_mmap+'_prob.npy', dtype=np.float32, mode='w+', shape=(args.dstore_size, 1))
                dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))
                dstore_tgts = np.memmap(args.dstore_mmap+'_tgts.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))
                dstore_src = np.memmap(args.dstore_mmap+'_src.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))

        if args.save_extra:
            writer = Writer(outdir='demo-out', max_size=args.save_extra_max_size, k=args.k, vec_size=1024)

        dstore_idx = 0
        dstore_full = False
        num_tokens = 0
        for ex_i, sample in tqdm(enumerate(t), desc='encode'):
            if 'net_input' not in sample:
                continue

            all_tokens = torch.cat([sample['net_input']['src_tokens'], sample['target'][:, -1, None]], -1)

            hf_batch = collections.defaultdict(list)
            for tok in all_tokens.tolist():
                tok = [tt for tt in tok if tt != dataset.pad_idx]
                raw_text = [task_dataset.vocab[tt] for tt in tok]
                hf_src_tokens, hf_target, hf_raw_target, hf_raw_text, hf_word_id, hf_mask = [], [], [], [], [], []
                for i_w in range(len(raw_text) - 1):
                    w = raw_text[i_w]
                    tok_ = hf_tokenizer.encode(w, add_special_tokens=False)
                    hf_src_tokens += tok_
                    hf_raw_text += hf_tokenizer.convert_ids_to_tokens(tok_)
                    hf_word_id += [i_w] * len(tok_)
                    hf_mask += [0] * (len(tok_) - 1) + [1]
                    hf_target += [tok[i_w + 1]]
                    hf_raw_target += [raw_text[i_w + 1]]

                hf_batch['src_tokens'].append(hf_src_tokens)
                hf_batch['target'].append(hf_target) # This is indexed by KNN-LM tokenizer.
                hf_batch['raw_target'].append(hf_raw_target)
                hf_batch['word_id'].append(hf_word_id)
                hf_batch['mask'].append(hf_mask)

                num_tokens += len(hf_src_tokens)

            if args.save_knnlm_dstore and not dstore_full:
                _keys = extra['keys']
                shape = _keys.shape
                if shape[0] == len(hypos) * args.tokens_per_sample or args.no_min_context:
                    if dstore_idx + shape[0] > args.dstore_size:
                        shape = [args.dstore_size - dstore_idx]
                        dstore_full = True
                    if args.dstore_fp16:
                        dstore_keys[dstore_idx:shape[0]+dstore_idx] = _keys[:shape[0]].view(
                            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
                        dstore_vals[dstore_idx:shape[0]+dstore_idx] = extra['target'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.int16)
                        dstore_prob[dstore_idx:shape[0]+dstore_idx] = extra['probs'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.float16)
                        dstore_tgts[dstore_idx:shape[0]+dstore_idx] = extra['target'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.int16)
                        dstore_src[dstore_idx:shape[0]+dstore_idx] = extra['src_tokens'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.int16)
                    else:
                        dstore_keys[dstore_idx:shape[0]+dstore_idx] = _keys[:shape[0]].view(
                            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float32)
                        dstore_vals[dstore_idx:shape[0]+dstore_idx] = extra['target'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.int)
                        dstore_prob[dstore_idx:shape[0]+dstore_idx] = extra['probs'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.float32)
                        dstore_tgts[dstore_idx:shape[0]+dstore_idx] = extra['target'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.int)
                        dstore_src[dstore_idx:shape[0]+dstore_idx] = extra['src_tokens'][:shape[0]].view(
                            -1, 1).cpu().numpy().astype(np.int)

                    dstore_idx += shape[0]
                else:
                    print('Skipping this one with shape', shape)
                if dstore_full:
                    print('Datastore is full with {} items.'.format(args.dstore_size))

            wps_meter.update(sample['ntokens'])
            t.log({'wps': round(wps_meter.avg)})

            # Write saved values to disk.
            if args.save_extra:
                writer.update(extra)

    if args.save_knnlm_dstore:
        print("dstore_idx", dstore_idx, "final shape", shape)
        print("Keys", dstore_keys.shape, dstore_keys.dtype)
        print("Vals", dstore_vals.shape, dstore_vals.dtype)

    logger.info('done with {} tokens'.format(num_tokens))


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
