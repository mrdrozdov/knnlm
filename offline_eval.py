import argparse
import collections
import os
import time

import faiss
import numpy as np
import torch

from tqdm import tqdm


_global = collections.defaultdict(list)


def main(args):
    dstore = Dstore.read(args.dstore, args.dstore_size)
    ppl = eval_ppl(dstore['p'])
    print('ppl = {}'.format(ppl))

    if args.knn:
        knn_dstore = Dstore.read(args.knn_dstore, args.knn_dstore_size, legacy=args.knn_legacy)
        indexfile = os.path.join(args.knn_dstore, 'knn.index')
        print('Build KNN...')
        knn = KNN(knn_dstore['vec'], knn_dstore['val'],
                  indexfile=indexfile,
                  probe=args.probe,
                  k=args.k,
                  sim_func=args.knn_sim_func,
                  lookup_mode=args.knn_lookup_mode,
                  )
        print('done.')

        writer = Writer(args.save_to, args.dstore_size, args.k)


        bsize = args.bsize
        dsize = dstore['vec'].shape[0]
        nbatches = dsize // bsize
        if nbatches * bsize < dsize:
            nbatches += 1

        new_p_lst = []

        # Build should do.
        should_do_lst = np.ones(nbatches)
        if args.ndownsample is not None:
            should_do_lst[:] = 0
            should_do_lst[:args.ndownsample] = 1
            np.random.shuffle(should_do_lst)

        should_do_lst = should_do_lst == 1

        out = collections.defaultdict(list)
        for i in tqdm(range(nbatches)):
            should_do = should_do_lst[i]
            if not should_do:
                print('skip {}'.format(i))
                continue
            start = i * bsize
            end = min(start + bsize, dsize)
            size = end - start

            xq = np.ones((size, 1024))
            xt = np.ones((size, 1))
            xp = np.ones((size, 1))

            xq[:] = dstore['vec'][start:end]
            xt[:] = dstore['tgt'][start:end]
            xp[:] = dstore['p'][start:end]

            knn_p, dist, knns = knn.get_knn_log_prob(xq, xt)

            if args.save:
                knn_tgts = knn_dstore['tgt'][knns.reshape(-1)].reshape(size, args.k, 1)[:]
                writer.update(dist, knns, knn_tgts, start)

            out['p'].append(xp)
            out['knn_p'].append(knn_p)
            out['dist'].append(dist)

        # DONE
        Timer.print_summary(_global)

        for k in list(out.keys()):
            out[k] = np.concatenate(out[k], axis=0)

        print('PERPLEXITY')
        p, knn_p, dist = [out[k] for k in ['p', 'knn_p', 'dist']]
        p = torch.from_numpy(p).float().cuda()
        knn_p = torch.from_numpy(knn_p).float().cuda()
        coeff_lst = np.arange(20) / 20
        for coeff in coeff_lst:
            if coeff == 0:
                new_p = p
            else:
                new_p = combine_knn_and_vocab_probs(
                            knn_p,
                            p,
                            coeff)
            ppl = eval_ppl(new_p)
            print('coeff = {:.3f}, knn_ppl = {}'.format(coeff, ppl))

        print('')



class Writer:
    def __init__(self, path, dstore_size=None, k=None):
        self.dstore_size = dstore_size
        self.k = k
        self.path = path
        self._initialized = False

    def initialize(self):
        print('Init writer...')
        path, dstore_size, k = self.path, self.dstore_size, self.k
        os.system('mkdir -p {}'.format(path))
        self.out = {}
        self.out['dist'] = np.memmap(os.path.join(path, 'lookup_dist.npy'), dtype=np.float32, mode='w+', shape=(dstore_size, k, 1))
        self.out['knns'] = np.memmap(os.path.join(path, 'lookup_knns.npy'), dtype=np.int, mode='w+', shape=(dstore_size, k, 1))
        self.out['knn_tgts'] = np.memmap(os.path.join(path, 'lookup_knn_tgts.npy'), dtype=np.int, mode='w+', shape=(dstore_size, k, 1))
        self.out['done'] = np.memmap(os.path.join(path, 'lookup_done.npy'), dtype=np.int, mode='w+', shape=(dstore_size, 1))
        self.out['done'][:] = 0 # defaults to 0
        self._initialized = True
        print('done.')

    def update(self, dist, knns, knn_tgts, offset):
        if not self._initialized:
            self.initialize()
        size = dist.shape[0]
        start = offset
        end = start + size
        self.out['dist'][start:end] = dist.reshape(size, self.k, 1)
        self.out['knns'][start:end] = knns.reshape(size, self.k, 1)
        self.out['knn_tgts'][start:end] = knn_tgts.reshape(size, self.k, 1)
        self.out['done'][start:end] = 1


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

    @staticmethod
    def print_summary(d):
        print('TIME SUMMARY')
        for k, lst in d.items():
            print('{} {}'.format(k, sum(lst)))


class Dstore:
    @staticmethod
    def read(path, dstore_size, vec_size=1024, legacy=False):
        dstore = {}
        dstore['vec'] = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(dstore_size, vec_size))
        dstore['val'] = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(dstore_size, 1))
        dstore['src'] = np.memmap(os.path.join(path, 'dstore_src.npy'), dtype=np.int, mode='r', shape=(dstore_size, 1))
        dstore['p'] = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(dstore_size, 1))
        dstore['tgt'] = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(dstore_size, 1))
        return dstore


class KNN:
    def __init__(self, keys, vals, indexfile=None, probe=None, k=None, sim_func=None, lookup_mode='default'):
        self.keys = keys
        self.vals = vals
        self.indexfile = indexfile
        self.k = k
        self.half = False
        self.lookup_mode = lookup_mode
        self.metric_type = 'l2'
        self.sim_func = sim_func
        self.index = self.setup_faiss()

    def setup_faiss(self):
        index = faiss.read_index(self.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        return index

    def get_knns(self, queries):
        dists, knns = self.index.search(queries, self.k)
        return dists, knns

    def lookup(self, keys, k, mode='default'):
        if mode == 'default':
            out = keys[k]
        elif mode == 'flat':
            out = keys[k.reshape(-1)]
        elif mode == 'sort':
            _k = k.reshape(-1)
            _k.sort()
            out = keys[_k]
        elif mode == 'unique':
            u, inv = np.unique(k, return_inverse=True)
            tmp = keys[u]
            out = tmp[inv]

        return out[:]

    def get_knn_log_prob(self, queries, tgt):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    with Timer() as t:
                        knns_vecs = torch.from_numpy(self.lookup(self.keys, k, self.lookup_mode)).float().cuda().view(qsize[0], self.k, -1)
                    _global['key_lookup'].append(t.interval)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs)**2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are BxC
        # reshape: BxC
        qshape = queries.shape
        queries = torch.from_numpy(queries).float().cuda().view(-1, qshape[-1])
        tgt = torch.from_numpy(tgt).long().cuda().view(-1)
        with Timer() as t:
            dists, knns = self.get_knns(queries.cpu().numpy())
        _global['index_search'].append(t.interval)
        # BxK
        dists = torch.from_numpy(dists).float().cuda()
        dists = dist_func(dists, knns, queries, function=self.sim_func)
        probs = torch.log_softmax(dists, dim=-1)

        with Timer() as t:
            index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt.unsqueeze(-1)).float()
        _global['val_lookup'].append(t.interval)
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone().cpu().numpy()

        # Bx1
        return yhat_knn_prob.reshape(qshape[0], 1), dists.cpu().numpy(), knns


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


def eval_ppl(p):
    avg_nll = -p.mean() / np.log(2)
    ppl = 2**avg_nll
    return ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # KNN
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--knn-legacy', action='store_true')
    parser.add_argument('--knn-dstore', default='./checkpoints_full', type=str)
    parser.add_argument('--knn-dstore-size', default=103225485, type=int)
    parser.add_argument('--probe', default=32, type=int)
    parser.add_argument('--k', default=1024, type=int)
    parser.add_argument('--knn-sim-func', default=None, type=str)
    parser.add_argument('--knn-lookup-mode', default='default', type=str)
    parser.add_argument('--bsize', default=100, type=int)
    parser.add_argument('--ndownsample', default=None, type=int)
    # Dstore
    parser.add_argument('--dstore', default='./dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    # Lookup
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-to', default=None, type=str)
    #
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(args)

    with torch.no_grad():
        main(args)


