export DATA_DIR=/
export TASK=chemprot
export OUTPUT_PATH=results/chemprot
export BERT_ALL_DIR=/cluster/home/it_stu114/PTMs/roberta_base

python -u sslreg.py \
    --task_name $TASK \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model roberta-base \
    --data_dir /cluster/home/it_stu114/nlp_datasets/chemprot \
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
    --num_train_epochs 3   \
    --has_test_label    \
    --save_best_model   \
    --force-overwrite