"""
Create a version of wikitext-103 by downsampling the train data.



TEXT=./downsampled_data/wt103-downsampled
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103-downsampled \
    --workers 20
"""

import argparse
import collections
import numpy as np
import os
import re


def main(args):
    np.random.seed(args.seed)

    tr_data = read_and_group(os.path.join(args.input, 'wiki.train.tokens'))
    _ = read_and_group(os.path.join(args.input, 'wiki.valid.tokens'))
    _ = read_and_group(os.path.join(args.input, 'wiki.test.tokens'))

    new_tr_data = read_and_downsample(os.path.join(args.input, 'wiki.train.tokens'), data=tr_data, n=100)
    os.system('mkdir -p {}'.format(args.output))
    write_data(os.path.join(args.output, 'wiki.train.tokens'), new_tr_data)

    _ = read_and_group(os.path.join(args.output, 'wiki.train.tokens'))

    print('Copying...')
    os.system('cp {} {}'.format(os.path.join(args.output, 'wiki.train.tokens'), os.path.join(args.output, 'wiki.valid.tokens')))
    os.system('cp {} {}'.format(os.path.join(args.output, 'wiki.train.tokens'), os.path.join(args.output, 'wiki.test.tokens')))
    # Copy the original train data to maintain a similar vocab size.
    os.system('cp {} {}'.format(os.path.join(args.input, 'wiki.train.tokens'), os.path.join(args.output, 'wiki.train.tokens')))


def write_data(path, data):
    print('Writing to... {}'.format(path))
    with open(path, 'w') as f:
        f.write('\n')
        for k, lst in data.items():
            for line in lst:
                f.write(line)


def read_and_downsample(path, data=None, n=100):
    if data is None:
        data = read_and_group(path)
    keys = list(data.keys())
    np.random.shuffle(keys)
    keys = keys[:n]
    new_data = {k: data[k] for k in keys}
    return new_data


def read_and_group(path):
    print('Reading... {}'.format(path))
    p = re.compile('^ = [^=]')
    data = collections.defaultdict(list)
    key = None
    with open(path) as f:
        for line in f:
            if p.match(line):
                key = line
            if key is None:
                continue
            data[key].append(line)
    print('Counting tokens...')
    with open(path) as f:
        wc = len(f.read().split())
    print('\tFound {} sections and about {} tokens.'.format(len(data), wc))
    return data




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./examples/language_model/wikitext-103', type=str)
    parser.add_argument('--output', default='./downsampled_data/wt103-downsampled', type=str)
    parser.add_argument('--seed', default=121, type=int)
    args = parser.parse_args()

    main(args)
