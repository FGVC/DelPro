import torch
import numpy as np
import dnnutil
import time
from pathlib import Path
from types import SimpleNamespace
import sys


DATA = Path('data/')


def setup(cfg, args):
    # get dataset and setup data loaders
    workers = cfg.hp.get('num_workers', 4)

    if 'collate' in cfg.other:
        collate = getattr(dataset, cfg.other['collate']['func'])
    else:
        collate = None

    data = cfg.data.data(**cfg.data.args)

    nsamp = cfg.hp.get('samples_per_epoch', None)
    loaders = dnnutil.get_dataloaders(data, args.batch_size,
                                      num_workers=workers,
                                      collate=collate,
                                      nsamp=nsamp)

    # setup the network
    net = dnnutil.load_model(cfg.model.model, args.model, **cfg.model.args)

    # setup the optimizer, loss function, and trainer
    params = (x for x in net.parameters() if x.requires_grad)
    optim = cfg.optim.optim(net.parameters(), **cfg.optim.args)
    if 'optim' in args:
        optim.load_state_dict(args.optim)
    loss_fn = cfg.loss.loss(**cfg.loss.args)
    trainer = cfg.trainer(net, optim, loss_fn)

    state = SimpleNamespace(net=net, loaders=loaders, optim=optim, trainer=trainer)
    return state


def main(commands=None, callback=None):
    parser = dnnutil.config_parser(run_dir='runs')
    args = parser.parse_args(args=commands)

    manager = dnnutil.ConfigManager(root=args.run_dir, run_num=args.rid,
                                    metric='accuracy')
    cfg = manager.setup(args)
    state = setup(cfg, args)

    t = time.asctime()
    in_args = ' '.join(sys.argv)
    print(t)
    manager.log_text(t)
    manager.log_text(in_args)
    print(f'Run {str(manager.run_dir)}')
    if 'config' in args:
        desc, deets = dnnutil.config_string(args.config)
        print(desc)
        print(deets)

    #schedule = dnnutil.EpochSetLR(state.optim, ((60, .1), (100, .1)), args.start - 1)

    for e in range(args.start, args.start + args.epochs):
        t = time.time()
        #schedule.step()
        state.trainer.train(state.loaders[0], e)
        state.trainer.eval(state.loaders[1], e)

        t = time.time() - t
        stats = state.trainer.get_stats()
        lr = state.optim.param_groups[-1]['lr']
        opt_state = state.optim.state_dict()
        manager.epoch_save(state.net, e, t, lr, *stats, optim=opt_state)


if __name__ == '__main__':
    main()

