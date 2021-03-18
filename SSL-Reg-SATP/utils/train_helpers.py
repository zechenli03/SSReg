#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/22 3:04 AM
# @Author: Zechen Li
# @File  : train_helpers.py
import torch
import torch.utils.data

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from glue.tasks import get_task
from utils.utils import *


def build_model(args, num_labels):
    print('Building net model...')

    config_net = AutoConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
    )

    # create the encoders
    net = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config=config_net
    )

    print('Building ssh model...')

    config_ssh = AutoConfig.from_pretrained(
        args.model,
        num_labels=args.auxiliary_labels,
    )

    ssh = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config=config_ssh
    )

    ssh.bert = net.bert
    head = ssh.classifier
    ssh = ssh.cuda()
    net = net.cuda()

    return net, head, ssh


def prepare_train_data(args):
    print('Preparing net training data...')

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    net_task = get_task(args.task_name, args.dataroot)

    net_examples = net_task.get_train_examples()

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

    print('Preparing ssh training data...')

    if args.auxiliary_labels == 2:
        ssh_task = get_task('aug-2', args.aug_dataroot)
    elif args.auxiliary_labels == 3:
        ssh_task = get_task('aug-3', args.aug_dataroot)
    else:
        ssh_task = get_task('aug-4', args.aug_dataroot)

    ssh_examples = ssh_task.get_train_examples()

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

    ssh_input_ids = torch.tensor(ssh_input_ids[:len(net_input_ids)])
    ssh_input_masks = torch.tensor(ssh_input_masks[:len(net_input_masks)])
    ssh_segment_ids = torch.tensor(ssh_segment_ids[:len(net_segment_ids)])
    ssh_label_ids = torch.tensor(ssh_label_ids[:len(net_label_ids)])

    trset = torch.utils.data.TensorDataset(net_input_ids, net_input_masks, net_segment_ids, net_label_ids,
                                           ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids)

    trset_sampler = torch.utils.data.RandomSampler(trset)
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, sampler=trset_sampler,
                                           num_workers=0)
    return trloader
