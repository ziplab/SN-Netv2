# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil
import warnings

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmcv.engine import collect_results_cpu, collect_results_gpu
from depth.apis import multi_gpu_test, single_gpu_test, multi_gpu_test_snnet
from depth.datasets import build_dataloader, build_dataset
from depth.models import build_depther

import numpy as np
import cv2
import time
from depth.datasets.pipelines import Compose
from mmcv.image import tensor2imgs
from PIL import Image
from collections import defaultdict
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--stitch-id', default=0, type=int, help='Stitch ID')
    args = parser.parse_args()

    # assert args.show or args.output_file, \
    #     'At least one output should be enabled.'


    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    pipeline = Compose(cfg.data.test.pipeline)
    # build the model and load checkpoint
    cfg.model.train_cfg = None

    model = build_depther(
        cfg.model,
        test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    model.eval()
    model.backbone.reset_stitch_id(1)

    def pre_pipeline(results):
        """Prepare results dict for pipeline."""
        pass
        # results['depth_fields'] = []
        # results['depth_scale'] = 1000
        #
        # # train/test share the same cam param
        # results['cam_intrinsic'] = \
        #     [[5.1885790117450188e+02, 0, 3.2558244941119034e+02],
        #      [5.1946961112127485e+02, 0, 2.5373616633400465e+02],
        #      [0                     , 0, 1                    ]]

    img_info = {'filename': args.img}
    img_data = dict(img_info=img_info)
    pre_pipeline(img_data)
    img_data = pipeline(img_data)
    img_data['img_metas'] = [[item.data] for item in img_data['img_metas']]
    img_data['img'] = [item[None,...] for item in img_data['img']]
    with torch.no_grad():
        result_depth = model(return_loss=False, **img_data)

    img_tensor = img_data['img'][0]
    img_metas = img_data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))


        model.show_result(
            img_show,
            result_depth,
            show=False,
            out_file='./test.jpg',
            format_only=False)


if __name__ == '__main__':
    main()
