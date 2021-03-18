#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/25 7:16 PM
# @Author: Zechen Li
# @File  : dev_helpers.py
import time
from utils.misc import *
from utils.train_helpers import *
from glue.evaluate import compute_metrics
from glue.tasks import get_task, MnliMismatchedProcessor
import torch.nn as nn


def prepare_test_data(args):
    print('Preparing net evaluating data...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    net_task = get_task(args.task_name, args.dataroot)

    net_examples = net_task.get_dev_examples()

    net_label_list = net_task.get_labels()
    net_label_map = {label: i for i, label in enumerate(net_label_list)}

    net_input_ids = []
    net_input_masks = []
    net_segment_ids = []
    net_label_ids = []

    for (ex_index, example) in enumerate(net_examples):
        net_input_id, net_input_mask, net_segment_id, net_label_id = \
            convert_example_to_feature(example, tokenizer, args.max_seq_length, net_label_map)
        net_input_ids.append(net_input_id)
        net_input_masks.append(net_input_mask)
        net_segment_ids.append(net_segment_id)
        net_label_ids.append(net_label_id)

    net_input_ids = torch.tensor(net_input_ids)
    net_input_masks = torch.tensor(net_input_masks)
    net_segment_ids = torch.tensor(net_segment_ids)
    net_label_ids = torch.tensor(net_label_ids)

    net_teset = torch.utils.data.TensorDataset(net_input_ids, net_input_masks, net_segment_ids, net_label_ids)
    net_sampler = torch.utils.data.SequentialSampler(net_teset)
    net_teloader = torch.utils.data.DataLoader(net_teset, batch_size=args.batch_size, sampler=net_sampler,
                                               num_workers=0)

    print('Preparing ssh evaluating data...')

    if args.auxiliary_labels == 2:
        ssh_task = get_task('aug-2', args.aug_dataroot)
    elif args.auxiliary_labels == 3:
        ssh_task = get_task('aug-3', args.aug_dataroot)
    else:
        ssh_task = get_task('aug-4', args.aug_dataroot)
        
    ssh_examples = ssh_task.get_dev_examples()

    ssh_label_list = ssh_task.get_labels()
    ssh_label_map = {label: i for i, label in enumerate(ssh_label_list)}

    ssh_input_ids = []
    ssh_input_masks = []
    ssh_segment_ids = []
    ssh_label_ids = []

    for (ex_index, example) in enumerate(ssh_examples):
        ssh_input_id, ssh_input_mask, ssh_segment_id, ssh_label_id = \
            convert_example_to_feature(example, tokenizer, args.max_seq_length, ssh_label_map)
        ssh_input_ids.append(ssh_input_id)
        ssh_input_masks.append(ssh_input_mask)
        ssh_segment_ids.append(ssh_segment_id)
        ssh_label_ids.append(ssh_label_id)

    ssh_input_ids = torch.tensor(ssh_input_ids)
    ssh_input_masks = torch.tensor(ssh_input_masks)
    ssh_segment_ids = torch.tensor(ssh_segment_ids)
    ssh_label_ids = torch.tensor(ssh_label_ids)

    ssh_teset = torch.utils.data.TensorDataset(ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids)

    ssh_sampler = torch.utils.data.SequentialSampler(ssh_teset)
    ssh_teloader = torch.utils.data.DataLoader(ssh_teset, batch_size=args.batch_size, sampler=ssh_sampler,
                                               num_workers=0)
    return net_teset, net_teloader, ssh_teset, ssh_teloader


def run_eval(teloader, model, print_freq, num_labels):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(teloader), batch_time, losses, prefix='Net Dev: ')

    all_logits = []
    all_labels = []
    end = time.time()

    for i, (input_ids, input_masks, segment_ids, label_ids) in enumerate(teloader):
        with torch.no_grad():
            input_ids, input_masks, segment_ids, label_ids = \
                input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(), label_ids.cuda()
            logits = model(input_ids=input_ids,
                           attention_mask=input_masks,
                           token_type_ids=segment_ids, )
            outputs = logits[0]
            if num_labels == 1:
                criterion = nn.MSELoss(reduction='none').cuda()
                tmp_eval_loss = criterion(outputs.view(-1), label_ids.view(-1))
            else:
                criterion = nn.CrossEntropyLoss(reduction='none').cuda()
                tmp_eval_loss = criterion(outputs.view(-1, num_labels), label_ids.view(-1))
            label = label_ids.cpu().numpy()

        logit = outputs.detach().cpu().numpy()
        all_logits.append(logit)
        all_labels.append(label)

        losses.update(tmp_eval_loss.mean().item(), len(label_ids))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(' * Net Evaluation loss {losses:.4e}'.format(losses=losses.avg))

    return all_logits, all_labels, losses.avg


def test_net(teloader, model, args, num_labels, print_freq=100):
    all_logits, all_labels, eval_loss = run_eval(teloader, model, print_freq, num_labels)

    task_name = args.task_name.lower()

    result = compute_task_metrics(task_name, all_logits, all_labels)

    print(' * Net Evaluation Acc is', result)

    result_metric = {
      "eval_loss": eval_loss,
      "metrics": result
    }


    if task_name == "mnli":
        mm_val_examples = MnliMismatchedProcessor().get_dev_examples(args.dataroot)

        mm_val_label_list = MnliMismatchedProcessor().get_labels()
        mm_val_label_map = {label: i for i, label in enumerate(mm_val_label_list)}

        mm_val_input_ids = []
        mm_val_input_masks = []
        mm_val_segment_ids = []
        mm_val_label_ids = []

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        for (ex_index, example) in enumerate(mm_val_examples):
            mm_val_input_id, mm_val_input_mask, mm_val_segment_id, mm_val_label_id = \
                convert_example_to_feature(example, tokenizer, args.max_seq_length, mm_val_label_map)
            mm_val_input_ids.append(mm_val_input_id)
            mm_val_input_masks.append(mm_val_input_mask)
            mm_val_segment_ids.append(mm_val_segment_id)
            mm_val_label_ids.append(mm_val_label_id)

        mm_val_input_ids = torch.tensor(mm_val_input_ids)
        mm_val_input_masks = torch.tensor(mm_val_input_masks)
        mm_val_segment_ids = torch.tensor(mm_val_segment_ids)
        mm_val_label_ids = torch.tensor(mm_val_label_ids)

        mm_val_teset = torch.utils.data.TensorDataset(mm_val_input_ids, mm_val_input_masks, mm_val_segment_ids,
                                                      mm_val_label_ids)
        mm_val_sampler = torch.utils.data.SequentialSampler(mm_val_teset)
        mm_val_teloader = torch.utils.data.DataLoader(mm_val_teset, batch_size=args.batch_size, sampler=mm_val_sampler,
                                                      num_workers=args.workers)

        mm_val_logits, mm_val_labels, mm_eval_loss = run_eval(mm_val_teloader, model, print_freq, num_labels)

        print(' * Mis-matched task:')
        mm_result = compute_task_metrics(task_name, mm_val_logits, mm_val_labels)
        print(' * Net Evaluation Acc is', mm_result)

        result_metric["mm_eval_loss"] = mm_eval_loss
        result_metric["mm_metrics"] = mm_result

    return result_metric        


def test_ssh(teloader, model, args):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(teloader), batch_time, losses, prefix='SSH Dev: ')

    all_logits = []
    all_labels = []
    end = time.time()

    for i, (input_ids, input_masks, segment_ids, label_ids) in enumerate(teloader):
        with torch.no_grad():
            input_ids, input_masks, segment_ids, label_ids = \
                input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(), label_ids.cuda()
            logits = model(input_ids=input_ids,
                           attention_mask=input_masks,
                           token_type_ids=segment_ids, )
            outputs = logits[0]
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()
            tmp_eval_loss = criterion(outputs.view(-1, args.auxiliary_labels), label_ids.view(-1))
            label = label_ids.cpu().numpy()

        logit = outputs.detach().cpu().numpy()
        all_logits.append(logit)
        all_labels.append(label)

        losses.update(tmp_eval_loss.mean().item(), len(label_ids))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(' * SSH Evaluation loss {losses:.4e}'.format(losses=losses.avg))

    task_name = 'aug'

    print(' * SSH Evaluation Acc is', compute_task_metrics(task_name, all_logits, all_labels))



def compute_task_metrics(task_name, logits, labels):
    if logits.shape[1] == 1:
        pred_arr = logits.reshape(-1)
    else:
        pred_arr = np.argmax(logits, axis=1)
    return compute_metrics(
        task_name=task_name,
        pred_srs=pred_arr,
        label_srs=labels,
    )
