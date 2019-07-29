import torch
import numpy as np
import dnnutil
import time
from pathlib import Path
from types import SimpleNamespace
import sys
import importlib
import tqdm
import dataset
import json


DATA = Path('data/')


def setup(args):
    # get dataset and setup data loaders
    #data_params = args.data.split(' ')
    #data_mod = importlib.import_module(data_params[0])
    #data_args = dict(x.split('=') for x in data_params[1:])
    #data = data_mod(**data_args)
    data = dataset.ReIDTestData(root='/multiview/datasets/amur-tigers/reid')

    loader = torch.utils.data.DataLoader(
                data,
                batch_size=32,
                num_workers=6,
                shuffle=False,
                pin_memory=True)

    # setup the network
    net_params = args.model.split(' ')
    mod, cls = net_params[0].split('.')
    net_mod = importlib.import_module(mod)
    net_cls = getattr(net_mod, cls)
    net_args = dict(x.split('=') for x in net_params[1:])
    net = dnnutil.load_model(net_cls, args.checkpoint, **net_args)

    return net, loader


def main():
    import argparse
    parser = argparse.ArgumentParser('Amur Tiger Test Set Evaluation')
    parser.add_argument('--model', default='models.ResnetEmbed',
        help='Specify the model and any necessary keyword args as '
             '"module.network_module kw1=val1 kw2=val2 ..."')
    # parser.add_argument('--data', default='dataset.ReIDDataset',
    #     help='Spedify the dataset and any necessary keyword args as '
    #          '"module.dataset_module kw1=val1 kw2=val2 ..."')
    parser.add_argument('-c', '--checkpoint', required=True,
        help='Path to the model checkpoint')
    args = parser.parse_args()

    t = time.asctime()
    in_args = ' '.join(sys.argv)
    print(t)
    print(in_args)

    net, loader = setup(args)
    net.eval()

    with torch.no_grad():
        embeddings = []
        ids = []
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
            img, id = batch
            img = img.cuda()
            emb = net(img)
            embeddings.append(emb)
            ids.append(id)
        embeddings = torch.cat(embeddings, 0)
        norm = embeddings.pow(2).sum(1)
        distance_matrix = (norm.view(1, -1) + norm -
                           2 * torch.mm(embeddings, embeddings.t()))
        mask = torch.eye(embeddings.size(0)).to(distance_matrix).byte()
        distance_matrix.masked_fill_(mask, 0)
        rank = distance_matrix.argsort(dim=1).cpu()
        ids = torch.cat(ids).cpu().tolist()

    res = []
    for i in range(rank.size(0)):
        q = ids[i]
        vals = [ids[k] for k in rank[i, 1:]]
        res.append({'query_id': q, 'ans_ids': vals})
    json.dump(res, open('test-results.json', 'w'))


if __name__ == '__main__':
    main()

