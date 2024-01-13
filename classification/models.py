# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

import utils
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

class ViTAnchor(VisionTransformer):
    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def extract_block_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        outs = {}

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            outs[i] = x.detach()
        return outs


    def selective_forward(self, x, begin, end):
        for i, blk in enumerate(self.blocks):
            if i < begin:
                continue
            if i > end:
                break
            x = blk(x)
        return x

    def forward_until(self, x, blk_id):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == blk_id:
                break

        return x

    def forward_from(self, x, blk_id):
        for i, blk in enumerate(self.blocks):
            if i < blk_id:
                continue
            x = blk(x)

        x = self.norm(x)
        x = self.forward_head(x)

        return x

    def forward_patch_embed(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        return x

    def forward_norm_head(self, x):
        x = self.norm(x)
        x = self.forward_head(x)
        return x


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

import os
@register_model
def mae_base_patch16(pretrained=False, pretrained_cfg=None, pretrained_deit=None, **kwargs):
    model = ViTAnchor(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg',
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if pretrained:
        checkpoint = torch.load(os.path.join(pretrained_deit, 'mae_pretrain_vit_base.pth'))
        info = model.load_state_dict(checkpoint["model"], strict=False)
        logger = utils.get_root_logger()
        logger.info(str(info))
    return model

@register_model
def mae_large_patch16(pretrained=False, pretrained_cfg=None,  pretrained_deit=None,  **kwargs):
    model = ViTAnchor(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, global_pool='avg',
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        checkpoint = torch.load(os.path.join(pretrained_deit, 'mae_pretrain_vit_large.pth'))
        info = model.load_state_dict(checkpoint["model"], strict=False)
        logger = utils.get_root_logger()
        logger.info(str(info))
    return model

@register_model
def mae_huge_patch14(pretrained=False, pretrained_cfg=None, pretrained_deit=None,  **kwargs):
    model = ViTAnchor(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True, global_pool='avg',
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        checkpoint = torch.load(os.path.join(pretrained_deit, 'mae_pretrain_vit_huge.pth'))
        info = model.load_state_dict(checkpoint["model"], strict=False)
        logger = utils.get_root_logger()
        logger.info(str(info))
    return model