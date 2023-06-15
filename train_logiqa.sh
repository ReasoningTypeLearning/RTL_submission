#!/bin/bash

schedule=$(echo 'increment')
model=$(echo 'chitanda/merit-roberta-large-v2')
path_checkpoint=$(echo 'checkpoint/')
name_checkpoint=$(echo 'merit_increment')

python main_train.py \
--path dataset/logiqa_data/ \
--layer_schedule $schedule \
--path_checkpoint $path_checkpoint \
--name_checkpoint $name_checkpoint \
--max_length 256 \
--num_layers 24 \
--model_name $model \
--batch_size 6 \
--grad_accumulate 4 \
--learning_rate 1e-5 \
--warmup_ratio 0.1 \
--task logiqa
