import numpy

from dstore_util import *


def npy_copy(x):
    out = np.empty_like(x)
    out[:] = x
    return out


class DatasetUtils:
    def build_context(self, args):
        if args.preset == 'test':
            dstore = Dstore(args.dstore, args.dstore_size, 1024)
            dstore.initialize(include_keys=True)
            dstore.add_neighbors(args.lookup, args.lookup_k)
            dstore.add_exact(args.lookup, args.lookup_k)

            train = {}
            train['p'] = npy_copy(dstore.prob[:])
            train['dist'] = -npy_copy(dstore.exact[:])
            train['tgts'] = npy_copy(dstore.tgts[:])
            train['knn_tgts'] = npy_copy(dstore.knn_tgts[:])
            train['knns'] = npy_copy(dstore.knns[:])
            train['keys'] = npy_copy(dstore.keys[:])

            dstore = Dstore(args.test_dstore, args.test_dstore_size, 1024)
            dstore.initialize(include_keys=True)
            dstore.add_neighbors(args.test_lookup, args.lookup_k)
            dstore.add_exact(args.test_lookup, args.lookup_k)

            test = {}
            test['p'] = npy_copy(dstore.prob[:])
            test['dist'] = -npy_copy(dstore.exact[:])
            test['tgts'] = npy_copy(dstore.tgts[:])
            test['knn_tgts'] = npy_copy(dstore.knn_tgts[:])
            test['knns'] = npy_copy(dstore.knns[:])
            test['keys'] = npy_copy(dstore.keys[:])

        elif args.preset == 'valid':
            dstore = Dstore(args.dstore, args.dstore_size, 1024)
            dstore.initialize(include_keys=True)
            dstore.add_neighbors(args.lookup, args.lookup_k)
            dstore.add_exact(args.lookup, args.lookup_k)

            p = npy_copy(dstore.prob[:])
            dist = -npy_copy(dstore.exact[:])
            tgts = npy_copy(dstore.tgts[:])
            knn_tgts = npy_copy(dstore.knn_tgts[:])
            knns = npy_copy(dstore.knns[:])
            keys = npy_copy(dstore.keys[:])

            limit = args.limit
            if limit > 0:
                p = p[:limit]
                dist = dist[:limit]
                tgts = tgts[:limit]
                knn_tgts = knn_tgts[:limit]
                knns = knns[:limit]
                keys = keys[:limit]

            train = {}
            train['p'] = p
            train['dist'] = dist
            train['tgts'] = tgts
            train['knn_tgts'] = knn_tgts
            train['knns'] = knns
            train['keys'] = keys

            test = train

        elif args.preset == 'cross_valid':
            dstore = Dstore(args.dstore, args.dstore_size, 1024)
            dstore.initialize(include_keys=True)
            dstore.add_neighbors(args.lookup, args.lookup_k)
            dstore.add_exact(args.lookup, args.lookup_k)

            p = npy_copy(dstore.prob[:])
            dist = -npy_copy(dstore.exact[:])
            tgts = npy_copy(dstore.tgts[:])
            knn_tgts = npy_copy(dstore.knn_tgts[:])
            knns = npy_copy(dstore.knns[:])
            keys = npy_copy(dstore.keys[:])

            limit = args.limit
            if limit > 0:
                p = p[:limit]
                dist = dist[:limit]
                tgts = tgts[:limit]
                knn_tgts = knn_tgts[:limit]
                knns = knns[:limit]
                keys = keys[:limit]

            piv = int(0.5 * tgts.shape[0])

            train = {}
            train['p'] = p[:piv]
            train['dist'] = dist[:piv]
            train['tgts'] = tgts[:piv]
            train['knn_tgts'] = knn_tgts[:piv]
            train['knns'] = knns[:piv]
            train['keys'] = keys[:piv]

            test = {}
            test['p'] = p[piv:]
            test['dist'] = dist[piv:]
            test['tgts'] = tgts[piv:]
            test['knn_tgts'] = knn_tgts[piv:]
            test['knns'] = knns[piv:]
            test['keys'] = keys[piv:]

        context = {}
        context['train'] = train
        context['test'] = test
        return context
