_base_ = [
    '../_base_/datasets/coco-stuff10k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SNNetv2',
        include_ls=True,
        include_sl=True,
        include_lsl=True,
        include_sls=True,
        lora_r=16,
        anchors=[
            dict(
                # type='MAE',
                pretrained='pretrained/deit_3_small_224_21k_mmseg.pth',
                img_size=(512, 512),
                patch_size=16,
                in_channels=3,
                embed_dims=384,
                num_layers=12,
                num_heads=6,
                mlp_ratio=4,
                init_values=1.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                norm_cfg=dict(type='LN', eps=1e-6),
                act_cfg=dict(type='GELU'),
                norm_eval=False,
                is_deit_3=True,  # new
                is_anchor=True,  # new
                out_indices=(4, 7, 9, 11)
            ),
            dict(
                # type='MAE',
                pretrained='pretrained/deit_3_large_224_21k_mmseg.pth',
                img_size=(512, 512),
                patch_size=16,
                in_channels=3,
                embed_dims=1024,
                num_layers=24,
                num_heads=16,
                mlp_ratio=4,
                init_values=1.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                norm_cfg=dict(type='LN', eps=1e-6),
                act_cfg=dict(type='GELU'),
                norm_eval=False,
                is_deit_3=True,  # new
                is_anchor=True,  # new
                out_indices=(9, 14, 19, 23)
            )
        ]
    ),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=1024,
        channels=256,
        in_index=3,
        num_classes=171,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        num_convs=1,
        up_scale=4,
        kernel_size=1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=171,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=171,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=171,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            up_scale=4,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(
    lr=0.01,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=True)

find_unused_parameters = True

custom_hooks = [dict(type='SNNetHook', priority='ABOVE_NORMAL')]