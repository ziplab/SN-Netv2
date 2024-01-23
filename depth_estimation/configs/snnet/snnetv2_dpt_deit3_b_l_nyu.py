_base_ = [
    '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    type='DepthEncoderDecoder',

    backbone=dict(
        type='SNNetv2',
        include_ls=True,
        include_sl=True,
        include_lsl=True,
        include_sls=True,
        lora_r=4,
        anchors=[
            dict(
                pretrained='pretrained/deit_3_base_224_21k_mmseg.pth',
                img_size=224,
                embed_dims=768,
                num_layers=12,
                num_heads=12,
                out_indices=(2, 5, 8, 11),
                final_norm=False,
                with_cls_token=True,
                output_cls_token=True,
                is_anchor=True,
            ),
            dict(
                pretrained='pretrained/deit_3_large_224_21k_mmseg.pth',
                img_size=224,
                embed_dims=1024,
                num_layers=24,
                num_heads=16,
                out_indices=(4, 11, 17, 23),
                final_norm=False,
                with_cls_token=True,
                output_cls_token=True,
                is_anchor=True,
            )]
        ),
    # pretrained='nfs/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth',
    decode_head=dict(
        type='DPTHead',
        in_channels=(1024, 1024, 1024, 1024),
        channels=256,
        embed_dims=1024,
        post_process_channels=[96, 192, 384, 768],
        readout_type='project',
        norm_cfg=None,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, warm_up=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
max_lr=1e-4 / 5
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
momentum_config = dict(
    policy='OneCycle'
)

evaluation = dict(interval=1)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size= (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(416, 544)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')),
        ])
]

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))


find_unused_parameters = True
