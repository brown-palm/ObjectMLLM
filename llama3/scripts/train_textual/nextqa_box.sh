#!/bin/bash

model_path={PATH_TO_LLAMA3_8B}/original

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--max_seq_len 1100 \
--batch_size 1 \
--accum_iter 8 \
--epochs 5 \
--warmup_epochs 1 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset nextqa \
--use_box \
--box_format textual \
--output_dir ./checkpoint/nextqa_box \
--project_name objectmllm_nextqa
