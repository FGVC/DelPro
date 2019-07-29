import json
import random
from collections import Counter
from itertools import groupby


def splits(train_size=75, nsplits=5, csv_file='data/reid_list_train.csv'):
    flist = open(csv_file).read().strip().split('\n')
    flist = list(map(lambda x: [x[0], x[1]], (x.split(',') for x in flist)))
    entity_counts = Counter([x[0] for x in flist])
    entities = list(entity_counts.keys())
    nclass = len(entity_counts)

    splits = {}
    for i in range(nsplits):
        s = random.sample(entities, nclass)
        splits[f'split{i+1}'] = dict(train=s[:train_size], test=s[train_size:])

    flist.sort()
    ent = {}
    for k, g in groupby(flist, lambda x: x[0]):
        ent[k] = [x[1] for x in g]
        splits['entities'] = ent

    return splits


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=75, 
                        help='Number of entities in training set')
    parser.add_argument('--nsplits', type=int, default=5,
                        help='Number of different splits to create')
    parser.add_argument('-o', '--out-file', type=str, default='splits.json',
                        help='Output json file')

    args = parser.parse_args()
    split = splits(args.train_size, args.nsplits)
    json.dump(split, open(args.out_file, 'w'))

