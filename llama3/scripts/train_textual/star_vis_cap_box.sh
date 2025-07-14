#!/bin/bash

model_path={PATH_TO_LLAMA3_8B}/original

torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--max_seq_len 1200 \
--batch_size 1 \
--accum_iter 8 \
--epochs 5 \
--warmup_epochs 1 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset star \
--use_cap \
--output_dir ./checkpoint/star_cap \
--project_name objectmllm_star


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./checkpoint/star_cap/checkpoint_best.pth \
--max_seq_len 2200 \
--batch_size 1 \
--accum_iter 8 \
--epochs 5 \
--warmup_epochs 1 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset star \
--use_cap \
--use_box \
--box_format textual \
--output_dir ./checkpoint/star_cap_box \
--project_name objectmllm_star


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./checkpoint/star_cap_box/checkpoint_best.pth \
--max_seq_len 2200 \
--batch_size 1 \
--accum_iter 8 \
--epochs 5 \
--warmup_epochs 1 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset star \
--use_vis \
--use_cap \
--use_box \
--box_format textual \
--output_dir ./checkpoint/star_vis_cap_box \
--project_name objectmllm_star