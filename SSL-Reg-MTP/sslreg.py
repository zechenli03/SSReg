import argparse
import json
import os
import pandas as pd

import logging

import torch.nn as nn

from datasets.tasks import get_task
from shared import model_setup as shared_model_setup
from ssl_reg.runners import ClassificationLMTaskRunner, RunnerParameters
import ssl_reg.model_setup as ssl_reg_model_setup
from language_modeling import model_setup as lm_model_setup
from pytorch_pretrained_bert.utils import at_most_one_of
import shared.initialization as initialization
import shared.log_info as log_info
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

# todo: cleanup imports


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # === Model parameters === #
    parser.add_argument("--bert_vocab_path", default=None, type=str)
    parser.add_argument("--bert_save_mode", default="all", type=str)
    parser.add_argument("--bert_all_dir", default=None, type=str)

    # === Other parameters === #
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_val",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_val_history",
                        action='store_true',
                        help="save performance for each epoch")
    parser.add_argument("--save_best_model",
                        action='store_true',
                        help="If set true, save the best model on the validation set. Else, save the final model.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classification_loss_weight",
                        default=1.,
                        type=float)
    parser.add_argument("--train_lm_loss_weight",
                        default=1.,
                        type=float)
    parser.add_argument("--adam_beta1",
                        default=0.9,
                        type=float)
    parser.add_argument("--adam_beta2",
                        default=0.98,
                        type=float)
    parser.add_argument("--clip_grad_norm", action="store_true")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.06,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--has_test_label",
                        action='store_true',
                        help="Do we have labels of test set on this dataset?")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--print-trainable-params', action="store_true")
    parser.add_argument('--not-verbose', action="store_true")
    parser.add_argument('--force-overwrite', action="store_true")
    args = parser.parse_args(*in_args)
    return args


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    log_info.print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)
    task = get_task(args.task_name, args.data_dir)


    tokenizer = AutoTokenizer.from_pretrained(args.bert_all_dir)
    
    classification_lm_model = ssl_reg_model_setup.MyBertClassificationLM(bert_load_path=args.bert_all_dir, num_labels=len(task.processor.get_labels()))

    if args.do_train:
        if args.print_trainable_params:
            print("TRAINABLE PARAMS:")
            print("  SHARED:")
            for param_name, param in classification_lm_model.classification_model.roberta.named_parameters():
                if param.requires_grad:
                    print("    {}  {}".format(param_name, tuple(param.shape)))
            print("  CLASSIFICATION:")
            for param_name, param in classification_lm_model.classification_model.named_parameters():
                if param.requires_grad and not param_name.startswith("roberta."):
                    print("    {}  {}".format(param_name, tuple(param.shape)))
            print("  LM:")
            for param_name, param in classification_lm_model.lm_model.named_parameters():
                if param.requires_grad and not param_name.startswith("roberta."):
                    print("    {}  {}".format(param_name, tuple(param.shape)))
        train_examples = task.get_train_examples()
        t_total = shared_model_setup.get_opt_train_steps(
            num_train_examples=len(train_examples),
            args=args,
        )
       
        parameters = list(classification_lm_model.classification_model.named_parameters()) + list(classification_lm_model.lm_model.lm_head.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in parameters if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in parameters if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_parameters,lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=1e-6, weight_decay=0.1) 
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_proportion * t_total,
                                            num_training_steps=t_total)
    else:
        train_examples = None
        t_total = 0
        optimizer = None

    runner = ClassificationLMTaskRunner(
        classification_lm_model=classification_lm_model,
        optimizer=optimizer,
        clip_grad_norm=args.clip_grad_norm,
        scheduler=scheduler,
        tokenizer=tokenizer,
        label_list=task.get_labels(),
        device=device,
        rparams=RunnerParameters(
            max_seq_length=args.max_seq_length,
            classification_loss_weight=args.classification_loss_weight, train_lm_loss_weight=args.train_lm_loss_weight, 
            learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
            t_total=t_total, warmup_proportion=args.warmup_proportion,
            num_train_epochs=args.num_train_epochs,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        ),
        output_path=args.output_dir
    )

    if args.do_train:
        if args.do_val_history:
            # for GLUE datasets, we do not have test set labels, it could only be evaluated by submitting to GLUE server.
            if not args.has_test_label:
                val_examples = task.get_dev_examples()
                results = runner.run_train_val(
                    train_examples=train_examples,
                    val_examples=val_examples,
                    task_name=task.name,
                )
                metrics_str = json.dumps(results, indent=2)
                with open(os.path.join(args.output_dir, "val_metrics_history.json"), "w") as f:
                    f.write(metrics_str)
            else:
                val_examples = task.get_dev_examples()
                test_examples = task.get_test_examples()
                results_val, results_test = runner.run_train_val_test(
                    train_examples=train_examples,
                    val_examples=val_examples,
                    test_examples=test_examples,
                    task_name=task.name,
                    save_best_model=args.save_best_model,
                )
                metrics_str = json.dumps(results_val, indent=2)
                with open(os.path.join(args.output_dir, "val_metrics_history.json"), "w") as f:
                    f.write(metrics_str)
                metrics_str = json.dumps(results_test, indent=2)
                with open(os.path.join(args.output_dir, "test_metrics_history.json"), "w") as f:
                    f.write(metrics_str)
        else:
            runner.run_train(train_examples, task_name=task.name)

    if args.do_save:
        if not args.save_best_model:
            # Save a trained model at the last epoch.
            ssl_reg_model_setup.save_bert(
                classification_lm_model=classification_lm_model,
                optimizer=optimizer, args=args,
                save_path=os.path.join(args.output_dir, "all_state.p"),
                save_mode=args.bert_save_mode,
            )

    if args.do_val:
        val_examples = task.get_dev_examples()
        runner.load_best_model(os.path.join(args.output_dir, "all_state.p"))
        results = runner.run_evaluate_with_label(val_examples, task_name=task.name, verbose=not args.not_verbose)
        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "val_preds.csv"), header=False, index=False)
        metrics_str = json.dumps({"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
            f.write(metrics_str)


    if args.do_test:
        test_examples = task.get_test_examples()
        runner.load_best_model(os.path.join(args.output_dir, "all_state.p"))
        results = runner.run_evaluate_with_label(test_examples, task_name=task.name, verbose=not args.not_verbose)
        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "test_preds.csv"), header=False, index=False)
        metrics_str = json.dumps({"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            f.write(metrics_str)
    


if __name__ == "__main__":
    main()
