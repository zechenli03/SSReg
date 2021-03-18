#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/22 2:16 AM
# @Author: Zechen Li
# @File  : main_restart.py

import argparse
import torch.backends.cudnn as cudnn
import logging
from utils.dev_helpers import *
from utils.train_helpers import *
from glue.tasks import glue_tasks_num_labels
from transformers import get_linear_schedule_with_warmup, AdamW
import utils.misc as misc
import collections as col


parser = argparse.ArgumentParser()
# GLUE tasks settings
parser.add_argument('--dataroot', default='./glue_data/SST-2/')
parser.add_argument('--aug_dataroot', default='./aug_data/SST-2/')
parser.add_argument('--task_name', default='sst')
########################################################################
# model settings
parser.add_argument('--model', default='bert-large-uncased')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--max_seq_length', default=128, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--weight_decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--warmup_proportion', default=0.1, type=float)
parser.add_argument('--adam_epsilon', default=1e-8, type=float)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--auxiliary_weight", default=0.1, type=float)
parser.add_argument("--auxiliary_labels", default=4, type=int)
parser.add_argument('--seed',
                    type=int,
                    default=-1,
                    help="random seed for initialization")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--do_save_every_epoch",
                    action='store_true')
parser.add_argument("--do_eval_ssl_task",
                    action='store_true')
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')
parser.add_argument('--force-overwrite', action="store_true")
########################################################################


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = parser.parse_args()
device, n_gpu = misc.init_cuda_from_args(args, logger=logger)
misc.init_seed(args, n_gpu=n_gpu, logger=logger)
misc.init_output_dir(args)
misc.save_args(args)
cudnn.benchmark = True

print_args(args)

tokenizer = AutoTokenizer.from_pretrained(args.model)

task_name = args.task_name.lower()
num_labels = glue_tasks_num_labels[task_name]
print(' * Now is dealing with', task_name, 'dataset, the number of labels is', num_labels)

net, head, ssh = build_model(args, num_labels)
trloader = prepare_train_data(args)
net_teset, net_teloader, ssh_teset, ssh_teloader = prepare_test_data(args)

num_training_steps = len(trloader) // args.gradient_accumulation_steps * args.epochs
parameters = list(net.named_parameters()) + list(head.named_parameters())
no_decay = ["bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in parameters if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [
            p for n, p in parameters if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_parameters, lr=args.lr, eps=args.adam_epsilon)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_proportion * num_training_steps,
                                            num_training_steps=num_training_steps)
if num_labels == 1:
    #  We are doing regression
    criterion = nn.MSELoss(reduction='none').cuda()
else:
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

criterion_ssh = nn.CrossEntropyLoss(reduction='none').cuda()


def train(trloader, epoch):
    net.train()
    ssh.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('NET Loss', ':.4e')
    losses_ssh = AverageMeter('SSH Loss', ':.4e')
    progress = ProgressMeter(len(trloader), batch_time, data_time, losses, losses_ssh,
                             prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, dl in enumerate(trloader):
        data_time.update(time.time() - end)

        net_input_ids, net_input_masks, net_segment_ids, net_label_ids = \
            dl[0].cuda(), dl[1].cuda(), dl[2].cuda(), dl[3].cuda()

        outputs_cls = net(input_ids=net_input_ids,
                          attention_mask=net_input_masks,
                          token_type_ids=net_segment_ids, )

        outputs_cls = outputs_cls[0]

        global num_labels
        if num_labels == 1:
            loss_cls = criterion(outputs_cls.view(-1), net_label_ids.view(-1))
        else:
            loss_cls = criterion(outputs_cls.view(-1, num_labels), net_label_ids.view(-1))
        loss_net = loss_cls.mean()

        losses.update(loss_net.item(), len(net_label_ids))

        if args.gradient_accumulation_steps > 1:
            loss_net = loss_net / args.gradient_accumulation_steps

        ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids = \
            dl[4].cuda(), dl[5].cuda(), dl[6].cuda(), dl[7].cuda()

        outputs_ssh = ssh(input_ids=ssh_input_ids,
                          attention_mask=ssh_input_masks,
                          token_type_ids=ssh_segment_ids, )

        outputs_ssh = outputs_ssh[0]
        loss_ssh = criterion_ssh(outputs_ssh.view(-1, args.auxiliary_labels), ssh_label_ids.view(-1))
        loss_ssh = loss_ssh.mean()

        losses_ssh.update(loss_ssh.item(), len(ssh_label_ids))

        if args.gradient_accumulation_steps > 1:
            loss_ssh = loss_ssh / args.gradient_accumulation_steps

        loss = loss_net + loss_ssh * args.auxiliary_weight

        loss.backward()

        if (i + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global global_step
            global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.gradient_accumulation_steps == 0 and \
                global_step % args.print_freq == 0:
            progress.print(i)
    print(' * Global step is', global_step)
    print(' * Net Training loss@1 {losses:.4e}'.format(losses=losses.avg))
    print(' * SSH Training loss@1 {losses:.4e}'.format(losses=losses_ssh.avg))
    print(' * Total Training loss@1 {losses:.4e}'.format(losses=losses.avg + losses_ssh.avg * args.auxiliary_weight))
    return losses.avg, losses.avg + losses_ssh.avg * args.auxiliary_weight


if args.resume:
    model_path = os.path.join(args.resume, "ckpt.pth")
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        ckpt = torch.load(model_path)
        global_step = ckpt['global_step']
        args.start_epoch = ckpt['epoch'] + 1
        epoch_result_dict = ckpt['epoch_result_dict']
        scheduler.load_state_dict(ckpt['scheduler'])
        net.load_state_dict(ckpt['net'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        test_net(net_teloader, net, args, num_labels, args.print_freq)
        print("=> loaded checkpoint '{}' (epoch {})".format(model_path, ckpt['epoch']))
    else:
        global_step = 0
        print("=> no checkpoint found at '{}'".format(model_path))
else:
    global_step = 0
    epoch_result_dict = col.OrderedDict()


for epoch in range(args.start_epoch, args.epochs + 1):
    net_train_loss, total_train_loss = train(trloader, epoch)
    epoch_result = test_net(net_teloader, net, args, num_labels, args.print_freq)
    if args.do_eval_ssl_task:
        test_ssh(ssh_teloader, ssh, args)
    epoch_result["net_train_loss"] = net_train_loss
    epoch_result["total_train_loss"] = total_train_loss
    epoch_result_dict[epoch] = epoch_result

    if args.do_save_every_epoch:
        if epoch < args.epochs:
            misc.save_results_history(epoch_result_dict, args)
            state = {'epoch': epoch,
                  'global_step': global_step,
                  'optimizer': optimizer.state_dict(),
                  'net': net.state_dict(),
                  'head': head.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'epoch_result_dict':epoch_result_dict}
            torch.save(state, args.outf + '/ckpt.pth')


misc.save_results_history(epoch_result_dict, args)

state = {'epoch': epoch,
      'global_step': global_step,
      'optimizer': optimizer.state_dict(),
      'net': net.state_dict(),
      'head': head.state_dict(),
      'scheduler': scheduler.state_dict(),
      'epoch_result_dict':epoch_result_dict}
torch.save(state, args.outf + '/ckpt.pth')

