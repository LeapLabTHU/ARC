_base_ = './orcnn_r50fpn1x_ss_dota10.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))