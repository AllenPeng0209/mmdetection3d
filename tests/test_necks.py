import torch

from mmdet3d.models.builder import build_backbone, build_neck


def test_centerpoint_rpn():
    second_cfg = dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False))

    second = build_backbone(second_cfg)

    centerpoint_fpn_cfg = dict(
        type='CenterPointFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False))

    centerpoint_fpn = build_neck(centerpoint_fpn_cfg)

    input = torch.rand([4, 64, 512, 512])
    sec_output = second(input)
    output = centerpoint_fpn(sec_output)
    assert output.shape == torch.Size([4, 384, 128, 128])
