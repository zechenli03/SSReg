#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/24 10:24 PM
# @Author: Zechen Li
# @File  : utils.py
import random


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "label": self.label,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def is_null_label_map(label_map):
    return len(label_map) == 1 and label_map[None] == 0


def convert_example_to_feature(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]

    return input_ids, input_mask, segment_ids, label_id


class TokenizedExample(object):
    def __init__(self, guid, tokens_a, tokens_b=None, label=None):
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.label = label

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "tokens_a": self.tokens_a,
            "tokens_b": self.tokens_b,
            "label": self.label,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


def tokenize_example(example, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    else:
        tokens_b = example.text_b
    return TokenizedExample(
        guid=example.guid,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        label=example.label,
    )


def convert_data_to_feature(df, tokenizer, max_seq_len):

    context_tokens = []
    context_loss_tokens = []

    for index, row in df.iterrows():
        context = row['sentence']
        conversion_context(context, tokenizer, context_tokens, context_loss_tokens, max_seq_len)

    max_seq_len = max_seq_len

    for c in context_tokens:
        while len(c) < max_seq_len:
            c.append(0)

    for c_l in context_loss_tokens:
        while len(c_l) < max_seq_len:
            c_l.append(-100)

    # BERT input embedding
    input_ids = context_tokens
    loss_ids = context_loss_tokens
    assert len(input_ids) == len(loss_ids)
    # data_features = {'input_ids': input_ids,
    #                  'loss_ids': loss_ids}

    return input_ids, loss_ids


def conversion_context(context, tokenizer, context_tokens, context_loss_tokens,max_seq_len):
    Mask_id_list = []
    new_word_piece_list = []

    while len(Mask_id_list) == 0:
        word_piece_list = tokenizer.tokenize(context)
        if len(word_piece_list) > max_seq_len - 2:
            word_piece_list = word_piece_list[:(max_seq_len - 2)]
        random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, new_word_piece_list)

    bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(new_word_piece_list))
    bert_loss_ids = []
    bert_loss_ids.append(-100)  # [CLS]:-100
    Mask_id_count = 0
    for word_piece in new_word_piece_list:
        if word_piece == '[MASK]':
            try:
                bert_loss_ids.append(Mask_id_list[Mask_id_count])
                Mask_id_count = Mask_id_count + 1
            except :
                print(new_word_piece_list)
                print(Mask_id_count)
                print(Mask_id_list)
                assert Mask_id_count < len(Mask_id_list)
        else:
            bert_loss_ids.append(-100)

    bert_loss_ids.append(-100)     # [SEP]:-100
    # print("bert_ids:",bert_ids)
    # print("bert_loss_ids:",bert_loss_ids)
    context_tokens.append(bert_ids)
    context_loss_tokens.append(bert_loss_ids)
    assert len(bert_ids) == len(bert_loss_ids)

    Mask_id_list.clear()
    new_word_piece_list.clear()
    word_piece_list.clear()


def random_change_word_piece(tokenizer, word_piece_list, Mask_id_list, new_word_piece_list):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    count = 0
    for word_piece in word_piece_list:
        if 0.15 >= random.random():
            change_probability = random.random()
            if change_probability <= 0.8:
                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                Mask_id_list.append(tokenizer.convert_tokens_to_ids(word_piece))
                new_word_piece_list.append('[MASK]')
            elif change_probability <= 0.9:
                # 10% of the time, we replace masked input tokens with random word
                vocab_index = random.randint(0, len(tokenizer.vocab)-1)
                new_word_piece_list.append(tokenizer.convert_ids_to_tokens(vocab_index))
            else:
                new_word_piece_list.append(word_piece)
        else:
            new_word_piece_list.append(word_piece)

    # Check whether the number of [MASK] is equal to the number of Mask_id_list. If it is not, the Mask_id_list is
    # cleared and then redone
    for new_word_piece in new_word_piece_list:
        if new_word_piece == '[MASK]':
            count = count + 1
    if count != len(Mask_id_list) or len(Mask_id_list) == count == 0:
        Mask_id_list.clear()
        new_word_piece_list.clear()
        word_piece_list.clear()
    # print("word_piece_list:",word_piece_list)
    # print("Mask_id_list:",Mask_id_list)
    # print("new_word_piece_list",new_word_piece_list)


