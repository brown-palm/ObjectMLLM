#!/bin/bash

model_path={PATH_TO_LLAMA3_8B}/original

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--max_seq_len 1100 \
--batch_size 1 \
--accum_iter 8 \
--epochs 10 \
--warmup_epochs 2 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset intentqa \
--use_cap \
--output_dir ./checkpoint/intentqa_cap \
--project_name objectmllm_intentqa


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./checkpoint/intentqa_cap/checkpoint_best.pth \
--max_seq_len 2100 \
--batch_size 1 \
--accum_iter 8 \
--epochs 10 \
--warmup_epochs 2 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset intentqa \
--use_cap \
--use_box \
--box_format textual \
--output_dir ./checkpoint/intentqa_cap_box \
--project_name objectmllm_intentqa


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./checkpoint/intentqa_cap_box/checkpoint_best.pth \
--max_seq_len 2100 \
--batch_size 1 \
--accum_iter 8 \
--epochs 10 \
--warmup_epochs 2 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset intentqa \
--use_vis \
--use_cap \
--use_box \
--box_format textual \
--output_dir ./checkpoint/intentqa_vis_cap_box \
--project_name objectmllm_intentqa