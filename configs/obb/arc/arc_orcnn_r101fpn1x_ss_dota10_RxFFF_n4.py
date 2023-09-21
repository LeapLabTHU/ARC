_base_ = [
    './orcnn_r101fpn1x_ss_dota10.py',
]

model = dict(
    pretrained='pretrained/ARC_ResNet101_xFFF_n4.pth',
    backbone=dict(
        type='ARCResNet',
        replace = [
            ['x'],
            ['0', '1', '2', '3'],
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'],
            ['0', '1', '2']
        ],
        kernel_number = 4,
    )
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5)
        }
    )
)