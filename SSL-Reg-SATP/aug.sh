for i in 'SST-2' 'CoLA' 'MRPC' 'QNLI' 'RTE' 'STS-B' 'WNLI' 'QQP' 'MNLI'
do
    echo Augment $i with two types ...
    python aug.py --num_type 2 --task_name $i --dataroot './glue_data/' --aug_dataroot './aug_data/type-2/' 
    echo Augment $i with three types ...
    python aug.py --num_type 3 --task_name $i --dataroot './glue_data/' --aug_dataroot './aug_data/type-3/' 
done

