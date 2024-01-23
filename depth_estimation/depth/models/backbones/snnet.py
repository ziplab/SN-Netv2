# Copyright (c) OpenMMLab. All rights reserved.import math
import json
import math

import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from depth.utils import get_root_logger
from ..builder import BACKBONES
from .vit import VisionTransformer
from mmcv.runner import BaseModule
import numpy as np
from mmcv.cnn import build_norm_layer
import torch.nn.functional as F
from typing import Optional, List
from collections import defaultdict
from .lora import wrap_model_with_lora, Linear, Conv2d
from .deit3 import DeiT3

def rearrange_activations(activations):
    n_channels = activations.shape[-1]
    activations = activations.reshape(-1, n_channels)
    return activations

def ps_inv(x1, x2):
    '''Least-squares solver given feature maps from two anchors.
    '''
    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)

    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when ' \
                         'calculating psuedo inverse matrix.')

    # Get transformation matrix shape
    shape = list(x1.shape)
    shape[-1] += 1

    # Calculate pseudo inverse
    x1_ones = torch.ones(shape)
    x1_ones[:, :-1] = x1
    A_ones = torch.matmul(torch.linalg.pinv(x1_ones), x2.to(x1_ones.device)).T

    # Get weights and bias
    w = A_ones[..., :-1]
    b = A_ones[..., -1]

    return w, b

def reset_out_indices(front_depth=12, end_depth=24, out_indices=(9, 14, 19, 23)):
    block_ids = torch.tensor(list(range(front_depth)))
    block_ids = block_ids[None, None, :].float()
    end_mapping_ids = torch.nn.functional.interpolate(block_ids, end_depth)
    end_mapping_ids = end_mapping_ids.squeeze().long().tolist()

    small_out_indices = []
    for i, idx in enumerate(end_mapping_ids):
        if i in out_indices:
            small_out_indices.append(idx)

    return small_out_indices


def get_stitch_configs_general_unequal(depths):
    depths = sorted(depths)

    total_configs = []

    # anchor configurations
    total_configs.append({'comb_id': [0], })
    total_configs.append({'comb_id': [1], })

    num_stitches = depths[0]
    for i, blk_id in enumerate(range(num_stitches)):
        if i == depths[0] - 1:
            break
        total_configs.append({
            'comb_id': (0, 1),
            'stitch_cfgs': (i, (i + 1) * (depths[1]//depths[0]))
        })
    return total_configs, num_stitches


def get_stitch_configs_bidirection(depths):
    depths = sorted(depths)

    total_configs = []

    # anchor configurations
    total_configs.append({'comb_id': [0], })
    total_configs.append({'comb_id': [1], })

    num_stitches = depths[0]

    # small --> large
    sl_configs = []
    for i, blk_id in enumerate(range(num_stitches)):
        sl_configs.append({
            'comb_id': [0, 1],
            'stitch_cfgs': [
                [i, (i + 1) * (depths[1] // depths[0])]
            ],
            'stitch_layer_ids': [i]
        })

    ls_configs = []
    lsl_confgs = []
    block_ids = torch.tensor(list(range(depths[0])))
    block_ids = block_ids[None, None, :].float()
    end_mapping_ids = torch.nn.functional.interpolate(block_ids, depths[1])
    end_mapping_ids = end_mapping_ids.squeeze().long().tolist()

    # large --> small
    for i in range(depths[1]):
        if depths[1] != depths[0]:
            if i % 2 == 1 and i < (depths[1] - 1):
                ls_configs.append({
                    'comb_id': [1, 0],
                    'stitch_cfgs': [[i, end_mapping_ids[i] + 1]],
                    'stitch_layer_ids': [i // (depths[1] // depths[0])]
                })
        else:
            if i < (depths[1] - 1):
                ls_configs.append({
                    'comb_id': [1, 0],
                    'stitch_cfgs': [[i, end_mapping_ids[i] + 1]],
                    'stitch_layer_ids': [i // (depths[1] // depths[0])]
                })

    # large --> small --> large
    for ls_cfg in ls_configs:
        for sl_cfg in sl_configs:
            if sl_cfg['stitch_layer_ids'][0] == depths[0] - 1:
                continue
            if sl_cfg['stitch_cfgs'][0][0] >= ls_cfg['stitch_cfgs'][0][1]:
                lsl_confgs.append({
                    'comb_id': [1, 0, 1],
                    'stitch_cfgs': [ls_cfg['stitch_cfgs'][0], sl_cfg['stitch_cfgs'][0]],
                    'stitch_layer_ids': ls_cfg['stitch_layer_ids'] + sl_cfg['stitch_layer_ids']
                })

    # small --> large --> small
    sls_configs = []
    for sl_cfg in sl_configs:
        for ls_cfg in ls_configs:
            if ls_cfg['stitch_cfgs'][0][0] >= sl_cfg['stitch_cfgs'][0][1]:
                sls_configs.append({
                    'comb_id': [0, 1, 0],
                    'stitch_cfgs': [sl_cfg['stitch_cfgs'][0], ls_cfg['stitch_cfgs'][0]],
                    'stitch_layer_ids': sl_cfg['stitch_layer_ids'] + ls_cfg['stitch_layer_ids']
                })

    total_configs += sl_configs + ls_configs + lsl_confgs + sls_configs

    anchor_ids = []
    sl_ids = []
    ls_ids = []
    lsl_ids = []
    sls_ids = []

    for i, cfg in enumerate(total_configs):
        comb_id = cfg['comb_id']

        if len(comb_id) == 1:
            anchor_ids.append(i)
            continue

        if len(comb_id) == 2:
            route = []
            front, end = cfg['stitch_cfgs'][0]
            route.append([0, front])
            route.append([end, depths[comb_id[-1]]])
            cfg['route'] = route
            if comb_id == [0, 1] and front != 11:
                sl_ids.append(i)
            elif comb_id == [1, 0]:
                ls_ids.append(i)

        if len(comb_id) == 3:
            route = []
            front_1, end_1 = cfg['stitch_cfgs'][0]
            front_2, end_2 = cfg['stitch_cfgs'][1]
            route.append([0, front_1])
            route.append([end_1, front_2])
            route.append([end_2, depths[comb_id[-1]]])
            cfg['route'] = route

            if comb_id == [1, 0, 1]:
                lsl_ids.append(i)
            elif comb_id == [0, 1, 0]:
                sls_ids.append(i)

        cfg['stitch_layer_ids'].append(-1)

    model_combos = [(0, 1), (1, 0)]
    return total_configs, model_combos, [len(sl_configs), len(ls_configs)], anchor_ids, sl_ids, ls_ids, lsl_ids, sls_ids


def format_out_features(outs, with_cls_token, hw_shape):
    if len(outs[0].shape) == 4:
        for i in range(len(outs)):
            outs[i] = outs[i].permute(0, 3, 1, 2).contiguous()
    else:
        B, _, C = outs[0].shape
        for i in range(len(outs)):
            if with_cls_token:
                # Remove class token and reshape token for decoder head
                temp = outs[i][:, 1:].reshape(B, hw_shape[0], hw_shape[1],
                                                 C).permute(0, 3, 1, 2).contiguous()
            else:
                temp = outs[i].reshape(B, hw_shape[0], hw_shape[1],
                                          C).permute(0, 3, 1, 2).contiguous()

            outs[i] = [temp, outs[i][:, 0]] # output cls token
    return outs


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# import loralib as lora


class StitchingLayer(BaseModule):
    def __init__(self, in_features=None, out_features=None, r=0):
        super().__init__()
        self.transform = Linear(in_features, out_features, r)

    def init_stitch_weights_bias(self, weight, bias):
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)

    def forward(self, x):
        out = self.transform(x)
        return out

@BACKBONES.register_module()
class SNNet(BaseModule):

    def __init__(self, anchors=None):
        super(SNNet, self).__init__()
        self.anchors = nn.ModuleList()
        for cfg in anchors:
            mod = DeiT3(**cfg)
            self.anchors.append(mod)

        self.with_cls_token = self.anchors[0].with_cls_token

        self.depths = [anc.num_layers for anc in self.anchors]

        # reset out indices of small
        self.anchors[0].out_indices = reset_out_indices(self.depths[0], self.depths[1], self.anchors[1].out_indices)

        total_configs, num_stitches = get_stitch_configs_general_unequal(self.depths)
        self.stitch_layers = nn.ModuleList(
            [StitchingLayer(self.anchors[0].embed_dims, self.anchors[1].embed_dims) for _ in range(num_stitches)])

        self.stitch_configs = {i: cfg for i, cfg in enumerate(total_configs)}
        self.all_cfgs = list(self.stitch_configs.keys())
        self.num_configs = len(total_configs)
        self.stitch_config_id = 0
        self.is_ranking = False

    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id

    def initialize_stitching_weights(self, x):
        logger = get_root_logger()
        front, end = 0, 1
        with torch.no_grad():
            front_features = self.anchors[front].extract_block_features(x)
            end_features = self.anchors[end].extract_block_features(x)

        for i, blk_id in enumerate(range(self.depths[0])):
            front_id, end_id = i, (i + 1) * (self.depths[1] // self.depths[0])
            front_blk_feat = front_features[front_id]
            end_blk_feat = end_features[end_id - 1]
            w, b = ps_inv(front_blk_feat, end_blk_feat)
            self.stitch_layers[i].init_stitch_weights_bias(w, b)
            logger.info(f'Initialized Stitching Model {front} to Model {end}, Layer {i}')

    def init_weights(self):
        for anc in self.anchors:
            anc.init_weights()


    def forward(self, x):

        # randomly sample a stitch at each training iteration
        if self.training:
            stitch_cfg_id = np.random.randint(0, self.num_configs)
        else:
            stitch_cfg_id = self.stitch_config_id

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']

        if len(comb_id) == 1:
            outs, hw_shape = self.anchors[comb_id[0]](x)

            # in case forwarding the smaller anchor
            if comb_id[0] == 0:
                for i, out_idx in enumerate(self.anchors[comb_id[0]].out_indices):
                    outs[i] = self.stitch_layers[out_idx](outs[i])

            outs = format_out_features(outs, self.with_cls_token, hw_shape)
            return outs

        cfg = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']

        x, outs, hw_shape = self.anchors[comb_id[0]].forward_until(x, blk_id=cfg[0])

        for i, out_idx in enumerate(self.anchors[comb_id[0]].out_indices):
            if out_idx < cfg[0]:
                outs[i] = self.stitch_layers[out_idx](outs[i])

        x = self.stitch_layers[cfg[0]](x)
        if cfg[0] in self.anchors[comb_id[0]].out_indices:
            outs[-1] = x

        B, _, C = x.shape

        outs_2 = self.anchors[comb_id[1]].forward_from(x, blk_id=cfg[1])

        outs += outs_2

        outs = format_out_features(outs, self.with_cls_token, hw_shape)

        return outs



@BACKBONES.register_module()
class SNNetv2(BaseModule):

    def __init__(self, anchors=None, selected_ids = [], include_sl=True, include_ls=True, include_lsl=True, include_sls=True, lora_r=0):
        super(SNNetv2, self).__init__()
        self.lora_r = lora_r

        self.anchors = nn.ModuleList()
        for cfg in anchors:
            mod = DeiT3(**cfg)
            self.anchors.append(mod)

        self.with_cls_token = self.anchors[0].with_cls_token
        # self.fix_stitching_layer = fix_stitch

        self.depths = [anc.num_layers for anc in self.anchors]

        # reset out indices of small
        self.anchors[0].out_indices = reset_out_indices(self.depths[0], self.depths[1], self.anchors[1].out_indices)
        total_configs, model_combos, num_stitches, anchor_ids, sl_ids, ls_ids, lsl_ids, sls_ids = get_stitch_configs_bidirection(self.depths)

        self.stitch_layers = nn.ModuleList()
        self.stitching_map_id = {}

        for i, (comb, num_sth) in enumerate(zip(model_combos, num_stitches)):
            front, end = comb
            temp = nn.ModuleList(
                [StitchingLayer(self.anchors[front].embed_dims, self.anchors[end].embed_dims, lora_r) for _ in range(num_sth)])
            temp.append(nn.Identity())
            self.stitch_layers.append(temp)

        self.stitch_configs = {i: cfg for i, cfg in enumerate(total_configs)}
        self.stitch_init_configs = {i: cfg for i, cfg in enumerate(total_configs) if len(cfg['comb_id']) == 2}

        self.selected_ids = selected_ids
        if len(selected_ids) == 0:
            self.all_cfgs = anchor_ids

            if include_sl:
                self.all_cfgs += sl_ids

            if include_ls:
                self.all_cfgs += ls_ids

            if include_lsl:
                self.all_cfgs += lsl_ids

            if include_sls:
                self.all_cfgs += sls_ids
        else:
            self.all_cfgs = selected_ids


        self.trained_cfgs = {}
        for idx in self.all_cfgs:
            self.trained_cfgs[idx] = self.stitch_configs[idx]

        logger = get_root_logger()
        logger.info(str(self.all_cfgs))
        self.num_configs = len(self.stitch_configs)
        self.stitch_config_id = 0


    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id

    def initialize_stitching_weights(self, x):
        logger = get_root_logger()
        anchor_features = []
        for anchor in self.anchors:
            with torch.no_grad():
                temp = anchor.extract_block_features(x)
                anchor_features.append(temp)

        for idx, cfg in self.stitch_init_configs.items():
            comb_id = cfg['comb_id']
            if len(comb_id) == 2:
                front_id, end_id = cfg['stitch_cfgs'][0]
                stitch_layer_id = cfg['stitch_layer_ids'][0]
                front_blk_feat = anchor_features[comb_id[0]][front_id]
                end_blk_feat = anchor_features[comb_id[1]][end_id - 1]
                w, b = ps_inv(front_blk_feat, end_blk_feat)
                self.stitch_layers[comb_id[0]][stitch_layer_id].init_stitch_weights_bias(w, b)
                logger.info(f'Initialized Stitching Layer {cfg}')


    def resize_abs_pos_embed(self, state_dict):
        pos_keys = [k for k in state_dict.keys() if 'pos_embed' in k]

        for pos_k in pos_keys:
            anchor_id = int(pos_k.split('.')[1])
        # if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict[pos_k]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = self.anchors[anchor_id].pos_embed.shape[-2] - self.anchors[anchor_id].num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int(
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
            # height (== width) for the new position embedding
            new_size = int(self.anchors[anchor_id].num_patches**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                                                    0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode=self.anchors[anchor_id].interpolate_mode,
                    align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                state_dict[pos_k] = new_pos_embed
        return state_dict

    def init_weights(self):
        for anc in self.anchors:
            anc.init_weights()

    def sampling_stitch_config(self):
        flops_id = np.random.choice(len(self.flops_grouped_cfgs))
        self.stitch_config_id = np.random.choice(self.flops_grouped_cfgs[flops_id])


    def get_stitch_parameters(self):
        stitch_cfg_id = self.stitch_config_id

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']

        total_params = 0

        # forward by a single anchor
        if len(comb_id) == 1:
            total_params += sum(p.numel() for p in self.anchors[comb_id[0]].parameters())
            if comb_id[0] == 0:
                for i, out_idx in enumerate(self.anchors[comb_id[0]].out_indices):
                    total_params += sum([p.numel() for p in self.stitch_layers[0][out_idx].parameters()])

            return total_params

        # forward among anchors
        route = self.stitch_configs[stitch_cfg_id]['route']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layer_ids']

        # patch embeding
        total_params += self.anchors[comb_id[0]].patch_embed_params()

        for i, (model_id, cfg) in enumerate(zip(comb_id, route)):
            total_params += self.anchors[model_id].selective_params(cfg[0], cfg[1])

            if model_id == 0:
                mapping_idx = [idx for idx in self.anchors[model_id].out_indices if cfg[0] <= idx <= cfg[1]]
                for j, out_idx in enumerate(mapping_idx):
                    total_params += sum([p.numel() for p in self.stitch_layers[model_id][out_idx].parameters()])

            total_params += sum([p.numel() for p in self.stitch_layers[model_id][stitch_layer_ids[i]].parameters()])

        return total_params

    def forward(self, x):

        if self.training:
            self.sampling_stitch_config()

        stitch_cfg_id = self.stitch_config_id

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']

        # forward by a single anchor
        if len(comb_id) == 1:
            outs, hw_shape = self.anchors[comb_id[0]](x)
            # in case forwarding the smaller anchor
            if comb_id[0] == 0:
                for i, out_idx in enumerate(self.anchors[comb_id[0]].out_indices):
                    outs[i] = self.stitch_layers[0][out_idx](outs[i])

            outs = format_out_features(outs, self.with_cls_token, hw_shape)
            return outs

        # forward among anchors
        route = self.stitch_configs[stitch_cfg_id]['route']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layer_ids']

        # patch embeding
        x, hw_shape = self.anchors[comb_id[0]].forward_patch_embed(x)
        final_outs = []

        for i, (model_id, cfg) in enumerate(zip(comb_id, route)):

            x, outs = self.anchors[model_id].selective_forward(x, cfg[0], cfg[1])

            if model_id == 0:
                mapping_idx = [idx for idx in self.anchors[model_id].out_indices if cfg[0] <= idx <= cfg[1]]
                for j, out_idx in enumerate(mapping_idx):
                    outs[j] = self.stitch_layers[model_id][out_idx](outs[j])

            final_outs += outs

            x = self.stitch_layers[model_id][stitch_layer_ids[i]](x)

        final_outs = format_out_features(final_outs, self.with_cls_token, hw_shape)

        return final_outs

