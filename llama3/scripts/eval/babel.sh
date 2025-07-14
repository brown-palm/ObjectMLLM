#!/bin/bash

model_path={PATH_TO_LLAMA3_8B}/original

# To evaluate our released checkpoints, please use:
CHECKPOINT_DIR=checkpoint_released

# To evaluate checkpoints reproduced by our training code, please use:
# CHECKPOINT_DIR=checkpoint


# Evaluate pose model
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 eval.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./$CHECKPOINT_DIR/babel_pose/checkpoint_best.pth \
--max_seq_len 1300 \
--batch_size 1 \
--dataset babel \
--use_pose \
--pose_format textual \
--output_dir ./$CHECKPOINT_DIR/babel_pose \
--project_name objectmllm_babel


# Evaluate video + pose model
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 eval.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./$CHECKPOINT_DIR/babel_vis_pose/checkpoint_best.pth \
--max_seq_len 1300 \
--batch_size 1 \
--dataset babel \
--use_vis \
--use_pose \
--pose_format textual \
--output_dir ./$CHECKPOINT_DIR/babel_vis_pose \
--project_name objectmllm_babel


# Evaluate caption + pose model
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 eval.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./$CHECKPOINT_DIR/babel_cap_pose/checkpoint_best.pth \
--max_seq_len 2100 \
--batch_size 1 \
--dataset babel \
--use_cap \
--cap_model gpt4o \
--use_pose \
--pose_format textual \
--output_dir ./$CHECKPOINT_DIR/babel_cap_pose \
--project_name objectmllm_babel