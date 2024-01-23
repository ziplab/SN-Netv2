# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from depth.datasets import build_dataloader, build_dataset
from depth.models import build_depther
from fvcore.nn import FlopCountAnalysis
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Depth benchmark a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    config_name = args.config.split('/')[-1].split('.')[0]
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=False,
    #     shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_depther(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    # model = MMDataParallel(model, device_ids=[0])

    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops = FlopCountAnalysis(model, torch.randn(1, 3, 480, 640).cuda()).total()

    print(flops)

if __name__ == '__main__':
    main()
