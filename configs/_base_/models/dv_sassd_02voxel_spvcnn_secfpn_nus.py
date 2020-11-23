voxel_size = [0.2, 0.2, 0.2]
model = dict(
    type='SASSD',
    pts_voxel_layer=dict(
        max_num_points=0, voxel_size=voxel_size, max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SPVCNN',
        in_channels=5,
        sparse_shape=[41, 512, 512],
        output_channels=64,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128],
        out_channels=[128, 128],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_points_head = dict(
        type='Points3DHead',
        num_classes=10,
        bbox_coder = dict(type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True),
        front_point_loss=dict(type='FocalLoss', reduction='mean', loss_weight=1.0),
        center_offset_loss=dict(type='SmoothL1Loss', reduction='mean', loss_weight=0.2)),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        seperate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True))
# model training and testing settings
train_cfg = dict(
    pts=dict(
        grid_size=[512, 512, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
test_cfg = dict(
    pts=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mod='spec', 
        score_thr=0.0,
        per_class_proposal=True, 
        max_output_num=100,
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        post_max_size=83,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2))