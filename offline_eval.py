import argparse
import collections
import os

import faiss
import numpy as np
import torch

from tqdm import tqdm


def main(args):
    dstore = Dstore.read(args.dstore, args.dstore_size)
    ppl = eval_ppl(dstore['p'])
    print('ppl = {}'.format(ppl))

    if args.knn:
        knn_dstore = Dstore.read(args.knn_dstore, args.knn_dstore_size, legacy=args.knn_legacy)
        indexfile = os.path.join(args.knn_dstore, 'knn.index')
        print('Build KNN...')
        knn = KNN(knn_dstore['vec'], knn_dstore['src'],
                  indexfile=indexfile,
                  probe=args.probe,
                  k=args.k,
                  sim_func=args.knn_sim_func,
                  )
        print('done.')

        coeff_lst = np.arange(20) / 20
        res = collections.defaultdict(list)

        bsize = 10000
        dsize = dstore['vec'].shape[0]
        nbatches = dsize // bsize
        if nbatches * bsize < dsize:
            nbatches += 1

        new_p_lst = []

        for i in tqdm(range(nbatches)):
            start = i * bsize
            end = min(start + bsize, dsize)
            size = end - start

            xq = np.ones((size, 1024))
            xt = np.ones((size, 1))
            xp = np.ones((size, 1))

            xq[:] = dstore['vec'][start:end]
            xt[:] = dstore['tgt'][start:end]
            xp[:] = dstore['p'][start:end]

            knn_p = knn.get_knn_log_prob(xq, xt)

            for coeff in coeff_lst:
                new_p = combine_knn_and_vocab_probs(
                            knn_p,
                            torch.from_numpy(xp).float().cuda(),
                            coeff)
                res[coeff].append(new_p.cpu().numpy())

            print('iter = {}'.format(i))

            for coeff in coeff_lst:
                new_p = np.concatenate(res[coeff], axis=0)
                ppl = eval_ppl(new_p)
                print('coeff = {:.3f}, knn_ppl = {}'.format(coeff, ppl))


class Dstore:
    @staticmethod
    def read(path, dstore_size, vec_size=1024, legacy=False):
        dstore = {}
        dstore['vec'] = np.memmap(os.path.join(path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(dstore_size, vec_size))
        dstore['src'] = np.memmap(os.path.join(path, 'dstore_vals.npy'), dtype=np.int, mode='r', shape=(dstore_size, 1))
        if not legacy:
            dstore['p'] = np.memmap(os.path.join(path, 'dstore_prob.npy'), dtype=np.float32, mode='r', shape=(dstore_size, 1))
            dstore['tgt'] = np.memmap(os.path.join(path, 'dstore_tgts.npy'), dtype=np.int, mode='r', shape=(dstore_size, 1))
        return dstore


class KNN:
    def __init__(self, keys, vals, indexfile=None, probe=None, k=None, sim_func=None):
        self.keys = keys
        self.vals = vals
        self.indexfile = indexfile
        self.k = k
        self.half = False
        self.metric_type = 'l2'
        self.sim_func = sim_func
        self.index = self.setup_faiss()

    def setup_faiss(self):
        index = faiss.read_index(self.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        return index

    def get_knns(self, queries):
        dists, knns = self.index.search(queries, self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    knns_vecs = torch.from_numpy(self.keys[k]).float().cuda().view(qsize[0], self.k, -1)
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
        dists, knns = self.get_knns(queries.cpu().numpy())
        # BxK
        dists = torch.from_numpy(dists).float().cuda()
        dists = dist_func(dists, knns, queries, function=self.sim_func)
        probs = torch.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()

        # Bx1
        return yhat_knn_prob.view(qshape[0], 1)


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
    parser.add_argument('--k', default=16, type=int)
    parser.add_argument('--knn-sim-func', default=None, type=str)
    # Dstore
    parser.add_argument('--dstore', default='./dstore_valid', type=str)
    parser.add_argument('--dstore-size', default=217646, type=int)
    args = parser.parse_args()

    with torch.no_grad():
        main(args)


