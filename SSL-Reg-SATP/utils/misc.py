#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/22 2:19 AM
# @Author: Zechen Li
# @File  : misc.py.py

import random
import numpy as np
import os
import torch
import json


def write_to_txt(name, content):
    with open(name, 'w') as text_file:
        text_file.write(content)


def mean(ls):
    return sum(ls) / len(ls)


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def print_nparams(model):
    nparams = sum([param.nelement() for param in model.parameters()])
    print('number of parameters: %d' % (nparams))


def normalize(v):
    return (v - v.mean()) / v.std()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_seed(seed):
    if seed == -1:
        return int(np.random.randint(0, 2**16 - 1))
    else:
        return seed


def init_seed(args, n_gpu, logger):
    if args.resume:
        if os.path.isfile(os.path.join(args.resume, "args.json")):
            with open(os.path.join(args.resume, "args.json")) as f:
                prev_args = json.load(f)
            seed = prev_args["seed"]
            logger.info("Setting to previous seed: {}".format(seed))
        else:
            seed = get_seed(args.seed)
            logger.info("Can not find previous args.json file. Randomly generate a number as new seed: {}".format(seed))
    else:
        seed = get_seed(args.seed)
        logger.info("Using seed: {}".format(seed))

    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    args.seed = seed

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_output_dir(args):
    if not args.force_overwrite \
            and (os.path.exists(args.outf) and os.listdir(args.outf)):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.outf))
    os.makedirs(args.outf, exist_ok=True)


def init_cuda_from_args(args, logger):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    return device, n_gpu

  
def save_args(args):
    with open(os.path.join(args.outf, "args.json"), "w") as f:
        f.write(json.dumps(vars(args), indent=2))

def save_results_history(results, args):
    metrics_str = json.dumps(results, indent=2)
    with open(os.path.join(args.outf, "results_metrics_history.json"), "w") as f:
        f.write(metrics_str)
