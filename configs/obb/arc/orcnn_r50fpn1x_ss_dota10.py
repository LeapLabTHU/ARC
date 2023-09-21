
_base_ = [
    '../oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py',
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)