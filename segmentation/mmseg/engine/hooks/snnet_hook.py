# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS
import torch
import json
import os

def group_subnets_by_flops(data, flops_step=10):
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}
    candidate_idx = []
    grouped_cands = []
    last_flops = 0
    for cfg_id, flops in sorted_data.items():
        # flops, _ = values
        flops = flops / 1e9
        if abs(last_flops - flops) > flops_step:
            if len(candidate_idx) > 0:
                grouped_cands.append(candidate_idx)
            candidate_idx = [int(cfg_id)]
            last_flops = flops
        else:
            candidate_idx.append(int(cfg_id))

    if len(candidate_idx) > 0:
        grouped_cands.append(candidate_idx)

    return grouped_cands



def initialize_model_stitching_layer(model, dataiter):
    images = []
    total_samples = 50
    while len(images) < total_samples:
        item = next(dataiter)
        data = model.data_preprocessor(item, True)
        images.extend(data['inputs'])

    images = torch.stack(images, dim=0)
    samples = images.cuda()
    model.backbone.initialize_stitching_weights(samples)

@HOOKS.register_module()
class SNNetHook(Hook):
    """Docstring for NewHook.
    """

    def before_train(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        if not runner._resume:
            initialize_model_stitching_layer(model, runner.train_loop.dataloader_iterator)

        # cfg = Config.fromfile(runner._cfg_file)
        cfg_name = runner.cfg.filename.split('/')[-1].split('.')[0]
        with open(os.path.join('./model_flops', f'snnet_flops_{cfg_name}.json'), 'r') as f:
            flops_params = json.load(f)

        flops_step = 10
        grouped_subnet = group_subnets_by_flops(flops_params, flops_step)
        model.backbone.flops_grouped_cfgs = grouped_subnet

