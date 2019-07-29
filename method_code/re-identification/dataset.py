import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import json
import random


# Keypoint order
# 0  lef tear
# 1  right ear
# 2  nose
# 3  right shoulder
# 4  right front paw
# 5  left shoulder
# 6  left front paw
# 7  right hip
# 8  right knee
# 9  right back paw
# 10 left hip
# 11 left knee
# 12 left back paw
# 13 root of tail
# 14 center, mid point of 3 and 14


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
    def __init__(self, root='data', split='reid-hard-splits.split1', color='RGB'):
        self.root = Path(root)
        splitfile, split = split.split('.')
        splits = json.load(self.root.joinpath(splitfile+'.json').open())
        self.split = splits[split]
        self.entity_files = splits['entities']

        self.keypoints = json.load(
                self.root.joinpath('reid_keypoints_train.json').open())
        self.flank_ids = {}
        for ent in self.split['test']:
            kp = self.keypoints[self.entity_files[ent][0]]
            flank = 1 if kp[6] < kp[39] else -1
            self.flank_ids[ent] = flank

        self.preprocess = basic_preprocess()
        self.color = color

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
        img = Image.open(self.data.root / 'train' / img).convert(self.data.color)
        img = self.preprocess(img)
        flank = self.data.flank_ids[lab]

        return img, int(lab), flank


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
            # sample a negative example
            cont_lab, c2 = random.sample(self.entities, 2)
            if cont_lab == anc_lab:
                cont_lab = c2
            flist = self.files[cont_lab]
            contrast = flist[random.randrange(0, len(flist))]

        root = self.data.root / 'train'
        anchor = Image.open(root / anchor).convert(self.data.color)
        anchor = self.preprocess(anchor)
        contrast = Image.open(root / contrast).convert(self.data.color)
        contrast = self.preprocess(contrast)

        return anchor, int(anc_lab), contrast, int(cont_lab)


class RankReIDDataset(ReIDDataset):
    def __init__(self, root='data', split='reid-hard-splits.split1', nc=30,
                 nimg=3, color='RGB'):
        super().__init__(root, split, color)
        self.nc = nc
        self.nimg = nimg

    def train(self):
        return _RankReIDData(self, 'train', self.nc, self.nimg)


class _RankReIDData(_ReIDData):
    def __init__(self, data, split, nc=30, nimg=3):
        super().__init__(data, split)
        self.nc = nc
        self.nimg = nimg
        self.root = self.data.root / 'train'
        self.sampler = EntitySampler(self, nc, 10)
        self.collate = concat_collate

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, ind):
        lab = self.entities[ind]
        files = self.files[lab]
        imgs = random.sample(files, self.nimg)
        imgs = [Image.open(self.root / i).convert(self.data.color) for i in imgs]
        imgs = torch.stack([self.preprocess(i) for i in imgs], 0)

        return imgs, [int(lab)] * self.nimg


class EntitySampler(torch.utils.data.Sampler):
    def __init__(self, data_source, n_ents, repeat=1):
        self.data_source = data_source
        self.n_ents = n_ents
        self.repeat = repeat

    def __len__(self):
        return self.n_ents * self.repeat

    def __iter__(self):
        return (x for _ in range(self.repeat) for x in
                random.sample(range(len(self.data_source)), self.n_ents))


from torch.utils.data.dataloader import default_collate
def concat_collate(batch):
    batch = default_collate(batch)
    img, label = batch
    shape = img.size()
    b, n, q = shape[0], shape[1], shape[2:]
    img = img.view(b * n, *q)
    label = torch.stack(label, 1).view(-1)

    return [img, label]

