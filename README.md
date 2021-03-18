# SSL-Reg

## 1. Introduction

This is a PyTorch implementation of our paper "Self-supervised Regularization for Text Classification".



## 2. Environment

The corresponding environments to our codes have been included as requirements.txt in the two folders.



## 3. SSL-Reg (MTP)

We provide example data format for ChemProt. As for other datasets, they are publicly available [here](https://github.com/allenai/dont-stop-pretraining/).

To finetune Roberta with SSL-Reg-MTP, use the command like below:

```bash
export DATA_DIR=/
export TASK=chemprot
export OUTPUT_PATH=results/chemprot
export BERT_ALL_DIR=/cluster/home/it_stu114/PTMs/roberta_base

python sslreg.py \
    --task_name $TASK \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model roberta-base \
    --data_dir nlp_datasets/chemprot \
    --bert_all_dir $BERT_ALL_DIR \
    --max_seq_length 512 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --train_lm_loss_weight 0.1 \
    --adam_beta1    0.9    \
    --adam_beta2    0.98   \
    --clip_grad_norm    \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 10   \
    --has_test_label    \
    --save_best_model   \
    --force-overwrite
```

Note that for different datasets, num_train_epochs and train_lm_loss_weight might be set to difference values.



## 4. SSL-Reg (SATP)

The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running any of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory `$GLUE_DIR`.

We used [EDA](https://github.com/jasonwei20/eda_nlp) as the sentence augmentation method.

To transform your GLUE task to sentence augmentation type prediction task, you can run [`aug.py`](https://github.com/Ryanro/SSL-Reg-SATP/blob/master/aug.py) with following scripts:

```bash
python aug.py \
  --task_name CoLA \
  --dataroot './glue_data/' \
  --aug_dataroot './aug_data/' 
```

or

```bash
for i in 'SST-2' 'CoLA' 'MRPC' 'QNLI' 'RTE' 'STS-B' 'WNLI' 'QQP' 'MNLI'
do
    echo Augment $i with two types ...
    python aug.py --num_type 2 --task_name $i --dataroot './glue_data/' --aug_dataroot './aug_data/type-2/' 
    echo Augment $i with three types ...
    python aug.py --num_type 3 --task_name $i --dataroot './glue_data/' --aug_dataroot './aug_data/type-3/' 
    echo Augment $i with four types ...
    python aug.py --num_type 4 --task_name $i --dataroot './glue_data/' --aug_dataroot './aug_data/type-4/' 
done
```

To finetune GLUE task with SSL-Reg-SATP, run [`main.py`](https://github.com/Ryanro/SSL-Reg-SATP/blob/master/main.py) with following scripts.

```bash
python main.py \
  --lr 3e-5 \
  --epochs 6 \
  --auxiliary_weight 0.4 \
  --max_seq_length 128 \
  --batch_size 8 \
  --do_eval_ssl_task \
  --gradient_accumulation_steps 4 \
  --dataroot ./glue_data/CoLA/ \
  --aug_dataroot ./glue_data/CoLA/aug_data/type-4/ \
  --auxiliary_labels 4 \
  --task_name cola --print_freq 10 \
  --force-overwrite \
  --outf results/results_CoLA/bert1
```

