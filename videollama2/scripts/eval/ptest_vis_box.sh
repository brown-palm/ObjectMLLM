#!/bin/bash

# To evaluate our released checkpoints, please use:
CHECKPOINT_DIR=checkpoint_released

# To evaluate checkpoints reproduced by our training code, please use:
# CHECKPOINT_DIR=checkpoint

YOUR_VIDEO_DIR={PATH_TO_PERCEPTION_TEST}/videos

python3 videollama2/eval/eval_mcqa.py \
--finetuned_path $CHECKPOINT_DIR/ptest_vis_box \
--num_frames 16 \
--dataset ptest_vis_box \
--video_folder $YOUR_VIDEO_DIR \
--num_workers 16