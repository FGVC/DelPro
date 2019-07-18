import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import json
import random


T = torchvision.transforms
totensor = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_img_list(csv_file='data/reid_list_train.csv'):
    data = open('reid_list_train.csv').read().strip().split('\n')
    data = [x.split(',') for x in data]
    images = [x[1] for x in data]
    labels = [int(x[0]) for x in data]

    return images, labels


def basic_preprocess():
    preprocess  = {
        'train': T.Compose([
            T.RandomResizedCrop(224, (0.5, 1.0)),
            T.ColorJitter(.2, .2, .2, .1),
            totensor,
        ]),
        'test': T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            totensor
        ])
    }

    return preprocess


class ClassifyDataset(object):
    def __init__(self, root='data'):
        self.root = Path(root)
        self.split = json.load(open(root + '/class-split.json'))

        T = torchvision.transforms
        self.augment = {
            'train': T.Compose([
                T.RandomResizedCrop(224, (0.5, 1.0)),
                T.ColorJitter(.2, .2, .2, .1),
            ]),
            'test': T.Compose([
                T.Resize(256),
                T.CenterCrop(224)
            ])
        }
        self.totensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train(self):
        return _ClassData(self, 'train')

    def test(self):
        return _ClassData(self, 'test')


class _ClassData(torch.utils.data.Dataset):
    def __init__(self, data, split='train'):
        self.data = data
        self.files = data.split[split]
        self.augment = data.augment[split]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind):
        img, label = self.files[ind]
        img = Image.open(self.data.root / 'train' / img).convert('RGB')
        img = self.augment(img)
        img = self.data.totensor(img)

        return img, label


class ReIDDataset(object):
    def __init__(self, root='data', split='split1'):
        self.root = Path(root)
        splits = json.load(self.root.joinpath('reid-splits.json').open())
        self.split = splits[split]
        self.entity_files = splits['entities']

        self.preprocess = basic_preprocess()

    def train(self):
        return _PairwiseReIDData(self, 'train')

    def test(self):
        return _ReIDData(self, 'test')


class _ReIDData(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.data = data
        self.entities = data.split[split]  # the entity labels for this split
        self.files = data.entity_files  # a dict from entity label to list of files
        self.file_index = [
            (x, lab) for lab in self.entities for x in self.files[lab]]
        self.preprocess = data.preprocess[split]

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, ind):
        img, lab = self.file_index[ind]
        img = Image.open(self.data.root / 'train' / img).convert('RGB')
        img = self.preprocess(img)

        return img, int(lab)


class _PairwiseReIDData(_ReIDData):
    def __init__(self, data, split):
        super().__init__(data, split)
        
    def __getitem__(self, ind):
        anchor, anc_lab = self.file_index[ind]

        # sample a contrast image
        if random.random() > 0.5:
            # sample a positive example
            contrast, c2 = random.sample(self.files[anc_lab], 2)
            if contrast == anchor:
                contrast = c2
            cont_lab = anc_lab
        else:
            cont_lab, c2 = random.sample(self.entities, 2)
            if cont_lab == anc_lab:
                cont_lab = c2
            flist = self.files[cont_lab]
            contrast = flist[random.randrange(0, len(flist))]

        anchor = Image.open(self.data.root / 'train' / anchor).convert('RGB')
        anchor = self.preprocess(anchor)
        contrast = Image.open(self.data.root / 'train' / contrast).convert('RGB')
        contrast = self.preprocess(contrast)

        return anchor, int(anc_lab), contrast, int(cont_lab)

