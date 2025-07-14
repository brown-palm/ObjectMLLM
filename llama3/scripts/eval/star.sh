#!/bin/bash

model_path={PATH_TO_LLAMA3_8B}/original

# To evaluate our released checkpoints, please use:
CHECKPOINT_DIR=checkpoint_released

# To evaluate checkpoints reproduced by our training code, please use:
# CHECKPOINT_DIR=checkpoint


# Evaluate box model with textual representation 
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 eval.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./$CHECKPOINT_DIR/star_box/checkpoint_best.pth \
--max_seq_len 1200 \
--batch_size 1 \
--dataset star \
--use_box \
--box_format textual \
--output_dir ./$CHECKPOINT_DIR/star_box \
--project_name objectmllm_star


# Evaluate caption + box model
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 eval.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./$CHECKPOINT_DIR/star_cap_box/checkpoint_best.pth \
--max_seq_len 2200 \
--batch_size 1 \
--dataset star \
--use_cap \
--use_box \
--box_format textual \
--output_dir ./$CHECKPOINT_DIR/star_cap_box \
--project_name objectmllm_star


# Evaluate video + caption + box model
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 1 eval.py \
--num_workers 16 \
--llama_model_path $model_path \
--resume ./$CHECKPOINT_DIR/star_vis_cap_box/checkpoint_best.pth \
--max_seq_len 2200 \
--batch_size 1 \
--dataset star \
--use_vis \
--use_cap \
--use_box \
--box_format textual \
--output_dir ./$CHECKPOINT_DIR/star_vis_cap_box \
--project_name objectmllm_star