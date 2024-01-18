# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_snnet
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator
import requests
import models
import models_v2

import utils
import time
import sys
import datetime
import os
from utils import get_root_logger
from snnet import SNNet, SNNetv2
from utils import find_top_candidates, group_subnets_by_flops
import warnings
warnings.filterwarnings("ignore")
from fvcore.nn import FlopCountAnalysis

from PIL import Image
import gradio as gr
import plotly.express as px

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--exp_name', default='deit', type=str, help='experiment name')
    parser.add_argument('--config', default=None, type=str, help='configuration')
    parser.add_argument('--scoring', action='store_true', default=False, help='configuration')
    parser.add_argument('--proxy', default='synflow', type=str, help='configuration')
    parser.add_argument('--snnet_name', default='snnetv2', type=str, help='configuration')
    parser.add_argument('--get_flops', action='store_true')
    parser.add_argument('--flops_sampling_k', default=None, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--low_rank', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--lora_r', default=64, type=int,
                        help='number of distributed processes')
    parser.add_argument('--flops_gap', default=1.0, type=float,
                        help='number of distributed processes')

    return parser

def initialize_model_stitching_layer(model, mixup_fn, data_loader,  device):
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            model.initialize_stitching_weights(samples)

        break

@torch.no_grad()
def analyse_flops_for_all(model, config_name):
    all_cfgs = model.all_cfgs
    stitch_results = {}

    for cfg_id in all_cfgs:
        model.reset_stitch_id(cfg_id)
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).cuda()).total()
        stitch_results[cfg_id] = flops

    save_dir = './model_flops'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, f'flops_{config_name}.json'), 'w+') as f:
        json.dump(stitch_results, f, indent=4)


def main(args):
    utils.init_distributed_mode(args)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(os.path.join(args.output_dir, f'{timestamp}.log'))

    logger.info(str(args))

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    from datasets import build_transform

    transform = build_transform(False, args)

    anchors = []
    for i, anchor_name in enumerate(args.anchors):
        logger.info(f"Creating model: {anchor_name}")
        anchor = create_model(
            anchor_name,
            pretrained=False,
            pretrained_deit=None,
            num_classes=1000,
            drop_path_rate=args.anchor_drop_path[i],
            img_size=args.input_size
        )
        anchors.append(anchor)


    model = SNNetv2(anchors, lora_r=args.lora_r)

    checkpoint = torch.load(args.resume, map_location='cpu')
    # torch.save({'model': checkpoint['model']}, './snnetv2_deit3_s_l_50ep.pth')

    logger.info(f"load checkpoint from {args.resume}")
    model.load_state_dict(checkpoint['model'])
            
    model.to(device)

    config_name = args.config.split('/')[-1].split('.')[0]
    model.eval()
    # analyse_flops_for_all(model, config_name)

    # set sampling probs for differetn FLOPs groups
    with open(os.path.join('./model_flops', f'flops_{config_name}.json'), 'r') as f:
        flops_params = json.load(f)


    eval_res = {}
    flops_res = {}
    with open('results/stitches_res_s_l.txt', 'r') as f:
        for line in f.readlines():
            epoch_stat = json.loads(line.strip())
            eval_res[epoch_stat['cfg_id']] = epoch_stat['acc1']
            flops_res[epoch_stat['cfg_id']] = epoch_stat['flops'] / 1e9


    def visualize_stitch_pos(stitch_id):
        if stitch_id == 13:
            # 13 is equivalent to 0
            stitch_id = 0

        names = [f'ID {key}' for key in flops_res.keys()]

        fig = px.scatter(x=flops_res.values(), y=eval_res.values(), hover_name=names)
        fig.update_layout(
            title=f"SN-Netv2 - Stitch ID - {stitch_id}",
            title_x=0.5,
            xaxis_title="GFLOPs",
            yaxis_title="mIoU",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            ),
            legend=dict(
                yanchor="bottom",
                y=0.99,
                xanchor="left",
                x=0.01),
        )
        # continent, DarkSlateGrey
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2)),
                          selector=dict(mode='markers'))

        fig.add_scatter(x=[flops_res[stitch_id]], y=[eval_res[stitch_id]], mode='markers', marker=dict(size=15), name='Current Stitch')
        return fig



    # Download human-readable labels for ImageNet.
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")

    def process_image(image, stitch_id):
        # inp = torch.from_numpy(image).permute(2, 0, 1).float()
        inp = transform(image).unsqueeze(0).to(device)
        model.reset_stitch_id(stitch_id)
        with torch.no_grad():
            prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
            confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
        fig = visualize_stitch_pos(stitch_id)
        return confidences, fig



    with gr.Blocks() as main_page:
        with gr.Column():
            gr.HTML("""
                <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; ">Stitched ViTs are Flexible Vision Backbones</h1>
                <div align="center"> <img align="center" src='file/.github/gradio_banner.png' width="70%"> </div>
                <h3 align="center" >This is the classification demo page of SN-Netv2, an flexible vision backbone that allows for 100+ runtime speed and performance trade-offs.</h3>
                <h3 align="center" >You can also run this gradio demo on your local GPUs at https://github.com/ziplab/SN-Netv2</h3>
                """)
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type='pil')
                    stitch_slider = gr.Slider(minimum=0, maximum=134, step=1, label="Stitch ID")
                    with gr.Row():
                        clear_button = gr.ClearButton()
                        submit_button = gr.Button()
                with gr.Column():
                    label_output = gr.Label(num_top_classes=5)
                    stitch_plot = gr.Plot(label='Stitch Position')


        submit_button.click(
            fn=process_image,
            inputs=[image_input, stitch_slider],
            outputs=[label_output, stitch_plot],
        )

        stitch_slider.change(
            fn=visualize_stitch_pos,
            inputs=[stitch_slider],
            outputs=[stitch_plot],
            show_progress=False
        )

        clear_button.click(
            lambda: [None, 0, None, None],
            outputs=[image_input, stitch_slider, label_output, stitch_plot],
        )

        gr.Examples(
            [
                ['./.github/demo.jpg', 0],
            ],
            inputs=[
                image_input,
                stitch_slider
            ],
            outputs=[
                label_output,
                stitch_plot
            ],
        )

    main_page.launch(allowed_paths=['./'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)

    output_dir = os.path.join('outputs', args.exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path) and not args.resume:
        setattr(args, 'resume', checkpoint_path)

    setattr(args, 'output_dir', output_dir)

    main(args)
