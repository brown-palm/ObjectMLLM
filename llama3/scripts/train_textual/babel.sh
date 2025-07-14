#!/bin/bash

model_path={PATH_TO_LLAMA3_8B}/original


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--max_seq_len 1300 \
--batch_size 1 \
--accum_iter 8 \
--epochs 50 \
--warmup_epochs 2 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset babel \
--use_pose \
--pose_format textual \
--output_dir ./checkpoint/babel_pose \
--project_name objectmllm_babel


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./checkpoint/babel_pose/checkpoint_best.pth \
--max_seq_len 1300 \
--batch_size 1 \
--accum_iter 8 \
--epochs 50 \
--warmup_epochs 2 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset babel \
--use_vis \
--use_pose \
--pose_format textual \
--output_dir ./checkpoint/babel_vis_pose \
--project_name objectmllm_babel


torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./checkpoint/babel_pose/checkpoint_best.pth \
--max_seq_len 2100 \
--batch_size 1 \
--accum_iter 8 \
--epochs 50 \
--warmup_epochs 2 \
--blr 9e-2 \
--weight_decay 0.14 \
--dataset babel \
--use_cap \
--cap_model gpt4o \
--use_pose \
--pose_format textual \
--output_dir ./checkpoint/babel_cap_pose \
--project_name objectmllm_babel