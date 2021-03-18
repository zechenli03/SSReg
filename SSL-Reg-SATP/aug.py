#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/10/15 10:19 PM
# @Author: Zechen Li
# @File  : aug.py.py
from glue.tasks import get_task
import pandas as pd
import numpy as np
import os
from augment.eda import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--num_aug', default=1, type=int)
parser.add_argument('--num_type', default=4, type=int)
parser.add_argument('--task_name', default='CoLA')
parser.add_argument('--dataroot', default='./glue_data/')
parser.add_argument('--aug_dataroot', default='./aug_data/')

args = parser.parse_args()

alpha = args.alpha
num_aug = args.num_aug
num_type = args.num_type
task_name = args.task_name
task_dir = os.path.join(args.dataroot, task_name)
task = get_task(task_name.lower(), task_dir)
output_dir = os.path.join(args.aug_dataroot, task_name)

try:
    os.makedirs(output_dir)
except OSError:
    pass

ori_train_df = task.get_train_df()
ori_dev_df = task.get_dev_df()

aug_train_df = pd.DataFrame(columns=["sentence", "label"])

print("Trainning dataset preview:")
print("train sentences num:", len(ori_train_df))
print("Original:", ori_train_df.head())

for i in ori_train_df.sentence:
    ori_train_sentence = i

    method_label = np.random.randint(0, num_type, 1)[0]
    method = augment_single_with_label(method_label)

    aug_train_sentences = eda(ori_train_sentence, alpha=alpha, num_aug=num_aug, method=method)
    for aug_sentence in aug_train_sentences:
        aug_train_df = aug_train_df.append({'sentence': aug_sentence, 'label': method}, ignore_index=True)

print("Augment:", aug_train_df.head())
print(aug_train_df['label'].value_counts(normalize=True) * 100)
aug_train_df.to_csv(os.path.join(output_dir, "train.tsv"), sep='\t', index=False)

print('---------------------------------------------------------')

aug_dev_df = pd.DataFrame(columns=["sentence", "label"])

print("Dev dataset preview:")
print("dev sentences num:", len(ori_dev_df))
print("Original:", ori_dev_df.head())

for i in ori_dev_df.sentence:
    ori_dev_sentence = i

    method_label = np.random.randint(0, num_type, 1)[0]
    method = augment_single_with_label(method_label)

    aug_dev_sentences = eda(ori_dev_sentence, alpha=alpha, num_aug=num_aug, method=method)
    for aug_sentence in aug_dev_sentences:
        aug_dev_df = aug_dev_df.append({'sentence': aug_sentence, 'label': method}, ignore_index=True)
        
print("Augment:", aug_dev_df.head())
print(aug_dev_df['label'].value_counts(normalize=True) * 100)
aug_dev_df.to_csv(os.path.join(output_dir, "dev.tsv"), sep='\t', index=False)

print("generated augmented sentences finished.")

