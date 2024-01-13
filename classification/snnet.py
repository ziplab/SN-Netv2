# Copyright (c) OpenMMLab. All rights reserved.import math
import json
import math

import torch
import torch.nn as nn

import numpy as np

from collections import defaultdict
from utils import get_root_logger
import torch.nn.functional as F

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
    total_configs.append({'comb_id': [1], })
    num_stitches = depths[0]
    for i, blk_id in enumerate(range(num_stitches)):
        total_configs.append({
            'comb_id': (0, 1),
            'stitch_cfgs': (i, (i + 1) * (depths[1] // depths[0]))
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
    B, _, C = outs[0].shape
    for i in range(len(outs)):
        if with_cls_token:
            # Remove class token and reshape token for decoder head
            outs[i] = outs[i][:, 1:].reshape(B, hw_shape[0], hw_shape[1],
                                             C).permute(0, 3, 1, 2).contiguous()
        else:
            outs[i] = outs[i].reshape(B, hw_shape[0], hw_shape[1],
                                      C).permute(0, 3, 1, 2).contiguous()
    return outs


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class StitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, r=0):
        super().__init__()
        self.transform = Linear(in_features, out_features,  r=r)

    def init_stitch_weights_bias(self, weight, bias):
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)

    def forward(self, x):
        out = self.transform(x)
        return out


class SNNet(nn.Module):

    def __init__(self, anchors=None):
        super(SNNet, self).__init__()
        self.anchors = nn.ModuleList(anchors)

        self.depths = [len(anc.blocks) for anc in self.anchors]

        total_configs, num_stitches = get_stitch_configs_general_unequal(self.depths)
        self.stitch_layers = nn.ModuleList(
            [StitchingLayer(self.anchors[0].embed_dim, self.anchors[1].embed_dim) for _ in range(num_stitches)])

        self.stitch_configs = {i: cfg for i, cfg in enumerate(total_configs)}
        self.all_cfgs = list(self.stitch_configs.keys())
        self.num_configs = len(self.all_cfgs)
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

    def sampling_stitch_config(self):
        self.stitch_config_id = np.random.choice(self.all_cfgs)

    def forward(self, x):

        stitch_cfg_id = self.stitch_config_id
        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']

        if len(comb_id) == 1:
            return self.anchors[comb_id[0]](x)

        cfg = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']

        x = self.anchors[comb_id[0]].forward_until(x, blk_id=cfg[0])
        x = self.stitch_layers[cfg[0]](x)
        x = self.anchors[comb_id[1]].forward_from(x, blk_id=cfg[1])

        return x


class SNNetv2(nn.Module):

    def __init__(self, anchors=None, include_sl=True, include_ls=True, include_lsl=True, include_sls=True, lora_r=0):
        super(SNNetv2, self).__init__()
        self.anchors = nn.ModuleList(anchors)

        self.lora_r = lora_r

        self.depths = [len(anc.blocks) for anc in self.anchors]

        total_configs, model_combos, num_stitches, anchor_ids, sl_ids, ls_ids, lsl_ids, sls_ids = get_stitch_configs_bidirection(self.depths)

        self.stitch_layers = nn.ModuleList()
        self.stitching_map_id = {}

        for i, (comb, num_sth) in enumerate(zip(model_combos, num_stitches)):
            front, end = comb
            temp = nn.ModuleList(
                [StitchingLayer(self.anchors[front].embed_dim, self.anchors[end].embed_dim, r=lora_r) for _ in range(num_sth)])
            temp.append(nn.Identity())
            self.stitch_layers.append(temp)

        self.stitch_configs = {i: cfg for i, cfg in enumerate(total_configs)}
        self.stitch_init_configs = {i: cfg for i, cfg in enumerate(total_configs) if len(cfg['comb_id']) == 2}


        self.all_cfgs = list(self.stitch_configs.keys())
        logger = get_root_logger()
        logger.info(str(self.all_cfgs))


        self.all_cfgs = anchor_ids

        if include_sl:
            self.all_cfgs += sl_ids

        if include_ls:
            self.all_cfgs += ls_ids

        if include_lsl:
            self.all_cfgs += lsl_ids

        if include_sls:
            self.all_cfgs += sls_ids

        self.num_configs = len(self.stitch_configs)
        self.stitch_config_id = 0

    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id

    def set_ranking_mode(self, ranking_mode):
        self.is_ranking = ranking_mode

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

    def init_weights(self):
        for anc in self.anchors:
            anc.init_weights()


    def sampling_stitch_config(self):
        flops_id = np.random.choice(len(self.flops_grouped_cfgs), p=self.flops_sampling_probs)
        stitch_config_id = np.random.choice(self.flops_grouped_cfgs[flops_id])
        return stitch_config_id

    def forward(self, x):

        if self.training:
            stitch_cfg_id = self.sampling_stitch_config()
        else:
            stitch_cfg_id = self.stitch_config_id

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']

        # forward by a single anchor
        if len(comb_id) == 1:
            return self.anchors[comb_id[0]](x)

        # forward among anchors
        route = self.stitch_configs[stitch_cfg_id]['route']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layer_ids']

        # patch embeding
        x = self.anchors[comb_id[0]].forward_patch_embed(x)

        for i, (model_id, cfg) in enumerate(zip(comb_id, route)):

            x = self.anchors[model_id].selective_forward(x, cfg[0], cfg[1])
            x = self.stitch_layers[model_id][stitch_layer_ids[i]](x)

        x = self.anchors[comb_id[-1]].forward_norm_head(x)
        return x

