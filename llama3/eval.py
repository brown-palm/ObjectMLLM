import os
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import val_one_epoch
from llama import Tokenizer_llama3
from llama_vqa import LLaMA_VQA
from dataloader import load_data


def get_args_parser():
    parser = argparse.ArgumentParser('Vamos training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default=None, type=str, help='path of llama model')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--video_max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')
    parser.add_argument('--box_max_feats', type=int, default=1, metavar='LENGTH', help='the maximum bounding box feature length')
    parser.add_argument('--box_input_dim', type=int, default=4, metavar='LENGTH', help='The vector length of bounding box. Only useful when use embedding projector for bounding boxes.')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=float, default=2, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='clevrer', type=str, help='dataset')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--project_name', default='nextqa', type=str, help='wandb project name')
    parser.add_argument('--exp_name', type=str, default=None, help='wandb experiment name')
    
    parser.add_argument('--bias', type=float, default=3.5, help='attention bias')
    parser.add_argument('--use_cap', action='store_true', help='use caption for NextQA')
    parser.add_argument('--use_vis', action='store_true', help='use visual features')
    parser.add_argument('--use_box', action='store_true', help='use bounding boxes')
    parser.add_argument('--use_pose', action='store_true', help='use human poses')
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help='precision')
    parser.add_argument('--proj_type', type=str, default='linear', help='projection type')
    parser.add_argument('--cap_model', type=str, default='llava_13b', help='caption model')
    parser.add_argument('--vis_model', type=str, default='clip', help='visual model')
    parser.add_argument('--box_format', type=str, default=None, help='box model')
    parser.add_argument('--pose_format', type=str, default=None, help='pose model')
    parser.add_argument('--zero_init', action='store_true', help='Zero initialize the embedding projectors')
    parser.add_argument('--dim_expand', default=1, type=int, help='Number of tokens to represent each bounding box when using embedding projector.')
    
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}/tokenizer.model')

    if 'intentqa' in args.dataset or 'babel' in args.dataset:
        data_loader_val = load_data(args, tokenizer, split='test')
    else:
        data_loader_val = load_data(args, tokenizer, split='val')

    model = LLaMA_VQA(args)
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    best_acc = 0.

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start evaluation")
    start_time = time.time()
    epoch = -1
    if args.distributed:
        data_loader_val.sampler.set_epoch(epoch)

    logits_dict = {}
    val_stats = val_one_epoch(model_without_ddp, data_loader_val, optimizer, epoch, logits_dict=logits_dict, use_vis=args.use_vis, args=args)
    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    torch.save(logits_dict, os.path.join(args.output_dir, 'logits_{}.pt'.format(misc.get_rank())))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.mm_used = []
    args.max_feats = {}
    args.input_dims = {}
    if args.use_vis:
        args.mm_used.append('video')
        args.max_feats['video'] = args.video_max_feats
        args.input_dims['video'] = {'clip': 768, 'dinov2': 1024, 'siglip':1152}[args.vis_model]
    if args.use_box:
        args.mm_used.append('box')
        args.max_feats['box'] = args.box_max_feats
        args.input_dims['box'] = args.box_input_dim
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
