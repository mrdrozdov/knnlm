import collections

import numpy as np
import torch

from tqdm import tqdm


class Model:
    def __init__(self, c, max_ngram=3):
        self.c = c
        self.c1_sum = sum(c[1].values())
        self.max_ngram = max_ngram

    def batch_predict(self, src, tgts):
        # TODO: Should we be able to use small max_ngram as desired?
        max_ngram = self.max_ngram
        assert src.shape == tgts.shape
        n = tgts.shape[0]
        p = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            tgt = tgts[i].item()
            context = tgts[i-max_ngram+1:i].flatten().tolist()
            p[i] = self.predict(context, tgt)
        return p

    def predict(self, context, tgt, coeff_list=[0.3, 0.4, 0.3]):
        p = 0
        max_ngram = len(context) + 1

        for size in range(1, max_ngram + 1):
            if size == 1:
                p_ = self.c[size][tgt] / self.c1_sum
            elif size - 1 > len(context):
                raise Exception('size = {}, len(context) = {}'.format(size, len(context)))
            else:
                top_key = tuple(context[-(size-1):] + [tgt])
                if size == 2:
                    bot_key = context[-1]
                else:
                    bot_key = tuple(context[-(size-1):])
                bot = self.c[size - 1][bot_key]
                if bot > 0:
                    p_ = self.c[size][top_key] / bot
                else:
                    p_ = 0

            coeff = coeff_list[-size]
            p += coeff * p_

        return p


def build_ngram_lm(trn_src, trn_tgt, limit=-1, max_ngram=3):

    n = trn_tgt.shape[0]
    if limit > 0:
        n = int(limit)

    c = {}

    for size in range(1, max_ngram + 1):
        c[size] = collections.Counter()

    for t in tqdm(range(100, n), desc='train'):

        win = trn_tgt[t-max_ngram+1:t+1].flatten().tolist()

        for size in range(1, max_ngram + 1):
            if size == 1:
                w_t = win[-1]
                c[size][w_t] += 1
            else:
                gram = tuple(win[-size:])
                c[size][gram] += 1

    model = Model(c=c, max_ngram=max_ngram)
    return model


if __name__ == '__main__':

    trn_tgt = np.random.randint(0, 100, 10000)
    trn_src = trn_tgt.copy()
    trn_src[1:] = trn_tgt[:-1]

    ngram_lm = build_ngram_lm(trn_src, trn_tgt, max_ngram=3)

    probs = ngram_lm.batch_predict(trn_src, trn_tgt)

    print(probs)

