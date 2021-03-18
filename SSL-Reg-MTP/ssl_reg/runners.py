import collections as col
import logging
import numpy as np
from tqdm import tqdm, trange
import os

import torch
import pytorch_pretrained_bert.utils as utils
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets.core import InputExample
from datasets.runners import (
    LabelModes, is_null_label_map,
    tokenize_example, truncate_seq_pair,
    warmup_linear, compute_task_metrics,
    get_label_mode,
)
from language_modeling.runners import random_word
from ssl_reg.core import Batch, InputFeatures

import copy


logger = logging.getLogger(__name__)


class TrainEpochState:
    def __init__(self):
        self.tr_loss = 0
        self.tr_classification_loss = 0
        self.tr_lm_loss = 0
        self.global_step = 0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0


def convert_example_to_features(example, tokenizer, max_seq_length, label_map, evaluate,
                                select_prob=0.15):
    if evaluate:
        select_prob = 0
    else:
        select_prob = 0.15
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

    tokens_a, t1_label = random_word(tokens_a, tokenizer, select_prob=select_prob)
    # We do not use tokens_b when doing masked lm
    # if tokens_b:
    #     tokens_b, t2_label = random_word(tokens_b, tokenizer, select_prob=select_prob)
    #     lm_label_ids = ([-100] + t1_label + [-100] + t2_label + [-100])
    # else:
    #     lm_label_ids = ([-100] + t1_label + [-100])
    lm_label_ids = ([-100] + t1_label + [-100])


    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids_lm = tokenizer.convert_tokens_to_ids(tokens)

    ################ For classification input, we should not mask tokens ###############
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

    # tokens_a, t1_label = random_word(tokens_a, tokenizer, select_prob=select_prob)
    # if tokens_b:
    #     # tokens_b, t2_label = random_word(tokens_b, tokenizer, select_prob=select_prob)
    #     lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    # else:
    #     lm_label_ids = ([-1] + t1_label + [-1])


    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids_classification = tokenizer.convert_tokens_to_ids(tokens)
    #################################################################


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids_classification)

    # Zero-pad up to the sequence length.
    while len(input_ids_classification) < max_seq_length:
        input_ids_classification.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    while len(input_ids_lm) < max_seq_length:
        input_ids_lm.append(0)
        lm_label_ids.append(-100)

    assert len(input_ids_classification) == max_seq_length
    assert len(input_ids_lm) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if is_null_label_map(label_map):
        classification_label_id = example.label
    else:
        classification_label_id = label_map[example.label]

    features = InputFeatures(
        guid=example.guid,
        input_ids_classification=input_ids_classification,
        input_ids_lm=input_ids_lm,
        input_mask=input_mask,
        segment_ids=segment_ids,
        classification_label_id=classification_label_id,
        lm_label_ids=lm_label_ids,
        tokens=tokens,
    )
    return features


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, evaluate=False, verbose=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_features(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
            evaluate=evaluate,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("lm input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids_lm]))
            logger.info("classification input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids_classification]))
            logger.info("mlm_labels: %s" % " ".join([str(x) for x in feature_instance.lm_label_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.classification_label_id))

        features.append(feature_instance)
    return features


def convert_to_dataset(features, label_mode):
    full_batch = features_to_data(features, label_mode=label_mode)
    if full_batch.classification_label_ids is None:
        dataset = TensorDataset(full_batch.input_ids_classification, full_batch.input_ids_lm, full_batch.input_mask,
                                full_batch.segment_ids)
    else:
        dataset = TensorDataset(full_batch.input_ids_classification, full_batch.input_ids_lm, full_batch.input_mask,
                                full_batch.segment_ids,
                                full_batch.classification_label_ids, full_batch.lm_label_ids)
    return dataset, full_batch.tokens


def features_to_data(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return Batch(
        input_ids_classification=torch.tensor([f.input_ids_classification for f in features], dtype=torch.long),
        input_ids_lm=torch.tensor([f.input_ids_lm for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        classification_label_ids=torch.tensor([f.classification_label_id for f in features], dtype=label_type),
        lm_label_ids=torch.tensor([f.lm_label_ids for f in features], dtype=torch.long),
        tokens=[f.tokens for f in features],
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 6:
                input_ids_classification, input_ids_lm, input_mask, segment_ids, classification_label_ids, lm_label_ids = batch
            elif len(batch) == 3:
                input_ids, input_mask, segment_ids = batch
                classification_label_ids, lm_label_ids = None, None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i+1) * batch_size]
            yield Batch(
                input_ids_classification=input_ids_classification,
                input_ids_lm=input_ids_lm,
                input_mask=input_mask,
                segment_ids=segment_ids,
                classification_label_ids=classification_label_ids,
                lm_label_ids=lm_label_ids,
                tokens=batch_tokens,
            )

    def __len__(self):
        return len(self.dataloader)


class RunnerParameters:
    def __init__(self, max_seq_length,
                 classification_loss_weight, train_lm_loss_weight,
                 learning_rate, gradient_accumulation_steps, t_total, warmup_proportion,
                 num_train_epochs, train_batch_size, eval_batch_size):
        self.max_seq_length = max_seq_length
        self.classification_loss_weight = classification_loss_weight
        self.train_lm_loss_weight = train_lm_loss_weight
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


class ClassificationLMTaskRunner:
    def __init__(self, classification_lm_model, optimizer, clip_grad_norm, scheduler, tokenizer, label_list, device, rparams, output_path=None):
        self.classification_lm_model = classification_lm_model
        self.classification_model = classification_lm_model.classification_model
        self.lm_model = classification_lm_model.lm_model
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams
        self.global_step = 0
        self.output_path = output_path
    
    def load_best_model(self, model_load_path):
        all_state = torch.load(model_load_path)
        self.classification_model.load_state_dict(all_state['model'])
        self.lm_model.load_state_dict(all_state['lm_model'])

    
    def save_best_model(self, save_mode="model_all", verbose=True):
        classification_model = self.classification_lm_model.classification_model
        lm_model = self.classification_lm_model.lm_model
        assert save_mode in [
            "all", "tunable", "model_all", "model_tunable",
        ]
        save_dict = dict()
        # Save model
        classification_model_to_save = classification_model.module \
            if hasattr(classification_model, 'module') \
            else classification_model  # Only save the model itself
        if save_mode in ["all", "model_all"]:
            classification_model_state_dict = classification_model_to_save.state_dict()
            lm_state_dict = lm_model.state_dict()
        elif save_mode in ["tunable", "model_tunable"]:
            raise NotImplementedError
        else:
            raise KeyError(save_mode)
        if verbose:
            print("Saving {} classification model elems:".format(len(classification_model_state_dict)))
            print("Saving {} lm model elems:".format(len(lm_state_dict)))
        save_dict["model"] = utils.to_cpu(classification_model_state_dict)
        save_dict["lm_model"] = utils.to_cpu(lm_state_dict)

        # Save optimizer
        if save_mode in ["all", "tunable"]:
            optimizer_state_dict = utils.to_cpu(self.optimizer.state_dict()) if self.optimizer is not None else None
            if verbose:
                print("Saving {} optimizer elems:".format(len(optimizer_state_dict)))

        torch.save(save_dict, os.path.join(self.output_path, "all_state.p"))

    def run_train(self, train_examples, task_name, verbose=True):
        if verbose:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader = self.get_train_dataloader(train_examples, batch_size=self.rparams.train_batch_size, verbose=verbose)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader, task_name)

    def run_train_val(self, train_examples, val_examples, task_name):
        epoch_result_dict = col.OrderedDict()
        for i in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            train_dataloader = self.get_train_dataloader(train_examples, batch_size=self.rparams.train_batch_size, verbose=True)
            self.run_train_epoch(train_dataloader, task_name)
            epoch_result = self.run_evaluate_with_label(val_examples, task_name, verbose=True)
            del epoch_result["logits"]
            epoch_result_dict[i] = epoch_result
        return epoch_result_dict

    def run_train_val_test(self, train_examples, val_examples, test_examples, task_name, save_best_model):
        epoch_result_dict_val = col.OrderedDict()
        epoch_result_dict_test = col.OrderedDict()
        best_f1 = -1
        for i in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            train_dataloader = self.get_train_dataloader(train_examples, batch_size=self.rparams.train_batch_size, verbose=True)
            self.run_train_epoch(train_dataloader, task_name)
            epoch_result_val = self.run_evaluate_with_label(val_examples, task_name, verbose=True)
            epoch_result_test = self.run_evaluate_with_label(test_examples, task_name, verbose=True)
            del epoch_result_val["logits"]
            del epoch_result_test["logits"]
            epoch_result_dict_val[i] = epoch_result_val
            epoch_result_dict_test[i] = epoch_result_test
            if save_best_model:
                if epoch_result_val["metrics"]["f1"] > best_f1:
                    best_f1 = epoch_result_val["metrics"]["f1"]
                    self.save_best_model()
        return epoch_result_dict_val, epoch_result_dict_test
    
    
    def run_train_epoch(self, train_dataloader, task_name, classification_loss=True):
        for _ in self.run_train_epoch_context(train_dataloader, task_name, classification_loss):
            del _
            pass

    def run_train_epoch_context(self, train_dataloader, task_name, classification_loss):
        self.classification_lm_model.train()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
                task_name=task_name,
                classification_loss=classification_loss,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state, task_name, classification_loss):
        batch = batch.to(self.device)
        classification_loss, lm_loss = self.classification_lm_model(
            input_ids_classification=batch.input_ids_classification,
            input_ids_lm=batch.input_ids_lm,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            classification_labels=batch.classification_label_ids,
            masked_lm_labels=batch.lm_label_ids,
            use_lm=True,
        )
        classification_loss = self.rparams.classification_loss_weight * classification_loss
        lm_loss = self.rparams.train_lm_loss_weight * lm_loss
        # loss = classification_loss
        if classification_loss:
            loss = classification_loss + lm_loss
        else:
            loss = lm_loss

        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        loss.backward()
        

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.tr_classification_loss += classification_loss.item()
        train_epoch_state.tr_lm_loss += lm_loss.item()
        print("[TRAIN] ")
        print("   GLUE: {:.4f}".format(train_epoch_state.tr_classification_loss/ (step+1)))
        print("     LM: {:.4f}".format(train_epoch_state.tr_lm_loss/ (step+1)))
        print("  TOTAL: {:.4f}".format(train_epoch_state.tr_loss/ (step+1)))


        train_epoch_state.nb_tr_examples += batch.input_ids_lm.size(0)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.classification_lm_model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

    def run_evaluate_with_label(self, val_examples, task_name, verbose=True):
        self.classification_lm_model.eval()
        val_dataloader = self.get_eval_dataloader(val_examples, verbose=verbose)
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        all_labels = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            with torch.no_grad():
                classification_loss = self.classification_model(
                    input_ids=batch.input_ids_classification,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_mask,
                    labels=batch.classification_label_ids,
                )[0]
                classification_logits = self.classification_model(
                    input_ids=batch.input_ids_classification,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_mask,
                )[0]
                label_ids = batch.classification_label_ids.cpu().numpy()

            classification_logits = classification_logits.detach().cpu().numpy()
            total_eval_loss += classification_loss.mean().item()

            nb_eval_examples += batch.input_ids_classification.size(0)
            nb_eval_steps += 1
            all_logits.append(classification_logits)
            all_labels.append(label_ids)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": compute_task_metrics(task_name, all_logits, all_labels),
        }

    def run_test_without_label(self, test_examples, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_examples, verbose=verbose)
        self.classification_lm_model.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Predictions (Test)")):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.classification_model(
                    input_ids=batch.input_ids_classification,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_mask,
                )[0]
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples, batch_size, ttt=False, verbose=True):
        train_features = convert_examples_to_features(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        train_data, train_tokens = convert_to_dataset(
            train_features, label_mode=get_label_mode(self.label_map),
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size,
        )
        return HybridLoader(train_dataloader, train_tokens)

    def get_eval_dataloader(self, eval_examples, batch_size=None, verbose=True):
        eval_features = convert_examples_to_features(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer, evaluate=True,
            verbose=verbose,
        )
        eval_data, eval_tokens = convert_to_dataset(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        if batch_size:
            eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=batch_size,
        )
        else:
            eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(eval_dataloader, eval_tokens)
